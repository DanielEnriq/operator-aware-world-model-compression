from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oawc.config import get_run_config
from oawc.models.lewm_loader import load_lewm_from_hf
from oawc.paths import OUTPUT_ROOT, phase1_dirs, save_json


def ensure_lewm_source_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    lewm_src = project_root / "external" / "le-wm"

    if not lewm_src.exists():
        raise FileNotFoundError(
            f"Missing LeWM source at {lewm_src}. "
            "Run: git clone https://github.com/lucas-maes/le-wm.git external/le-wm"
        )

    if str(lewm_src) not in sys.path:
        sys.path.insert(0, str(lewm_src))


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def make_action_blocks(actions: np.ndarray, block_size: int) -> np.ndarray:
    """
    actions: (T, E, A)
    returns: (T - block_size + 1, E, block_size * A)
    """
    T, E, A = actions.shape
    blocks = []

    for t in range(T - block_size + 1):
        block = actions[t : t + block_size]       # (B, E, A)
        block = np.transpose(block, (1, 0, 2))    # (E, B, A)
        block = block.reshape(E, block_size * A)
        blocks.append(block)

    return np.asarray(blocks, dtype=np.float32)


def compute_action_block_stats(
    actions: np.ndarray,
    action_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_blocks = make_action_blocks(actions, action_block)
    flat = action_blocks.reshape(-1, action_blocks.shape[-1])

    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)

    return mean, std


def normalize_action_blocks(
    action_blocks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((action_blocks - mean) / (std + 1e-6)).astype(np.float32)


def to_pixel_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    """
    x: (B, T, H, W, C), uint8
    returns: (B, T, C, H, W), ImageNet-normalized
    """
    t = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    t = t.permute(0, 1, 4, 2, 3).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    return (t - mean) / std


def load_model(checkpoint_repo: str, device: str, model_path: str | None) -> torch.nn.Module:
    ensure_lewm_source_on_path()

    if model_path is None:
        model = load_lewm_from_hf(checkpoint_repo, device=device)
    else:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()

    return model


def sample_contexts(
    obs: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    *,
    history_size: int,
    action_block: int,
    rollout_blocks: int,
    num_contexts: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Builds planning-style examples.

    Each example has:
      context pixels: H frames
      goal pixel: future frame after H + rollout_blocks blocks
      context+future true action blocks: H + rollout_blocks blocks

    This is not environment-success evaluation. It is action-candidate cost ranking
    fidelity under the LeWM latent planning cost.
    """
    T, E = actions.shape[:2]
    done = np.logical_or(terminateds, truncateds)
    action_blocks = make_action_blocks(actions, action_block)

    valid = []
    max_start = T - (history_size + rollout_blocks) * action_block

    for t in range(max_start):
        final_time = t + (history_size + rollout_blocks) * action_block
        for e in range(E):
            if done[t : final_time + 1, e].any():
                continue
            valid.append((t, e))

    if len(valid) == 0:
        raise RuntimeError("No valid ranking-eval contexts found.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid), size=min(num_contexts, len(valid)), replace=False)

    context_pixels = []
    goal_pixels = []
    true_action_sequences = []

    for idx in chosen:
        t, e = valid[int(idx)]

        frame_times = [t + k * action_block for k in range(history_size)]
        goal_time = t + (history_size + rollout_blocks) * action_block
        action_times = [t + k * action_block for k in range(history_size + rollout_blocks)]

        context_pixels.append(obs[frame_times, e])
        goal_pixels.append(obs[goal_time, e])
        true_action_sequences.append(action_blocks[action_times, e])

    return {
        "context_pixels": np.asarray(context_pixels, dtype=np.uint8),
        "goal_pixels": np.asarray(goal_pixels, dtype=np.uint8),
        "true_action_sequences": np.asarray(true_action_sequences, dtype=np.float32),
    }


def make_candidate_actions(
    true_action_sequences: np.ndarray,
    *,
    num_candidates: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """
    true_action_sequences: (B, T_plan, A_block)
    returns: (B, S, T_plan, A_block)

    Candidate 0 is the true action sequence.
    Other candidates are Gaussian perturbations around it.
    """
    rng = np.random.default_rng(seed)

    B, T_plan, A_block = true_action_sequences.shape
    candidates = np.repeat(true_action_sequences[:, None], num_candidates, axis=1)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(B, num_candidates, T_plan, A_block),
    ).astype(np.float32)

    noise[:, 0] = 0.0
    candidates = candidates + noise

    return candidates.astype(np.float32)


def rollout_costs(
    model: torch.nn.Module,
    *,
    context_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    candidate_actions: np.ndarray,
    history_size: int,
    device: str,
) -> tuple[np.ndarray, float, int | None]:
    """
    Compute terminal latent costs for candidate action sequences.

    context_pixels:    (B, H, H_img, W_img, C)
    goal_pixels:       (B, H_img, W_img, C)
    candidate_actions: (B, S, T_plan, A_block), already normalized

    returns:
      costs: (B, S)
      elapsed_sec
      max_cuda_memory_bytes or None
    """
    model.eval()

    B, S, T_plan, A_block = candidate_actions.shape
    H = history_size

    if T_plan <= H:
        raise ValueError(f"T_plan={T_plan} must exceed history_size={H}")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.no_grad():
        ctx = to_pixel_tensor(context_pixels, device)  # (B,H,C,h,w)
        goal = to_pixel_tensor(goal_pixels[:, None], device)  # (B,1,C,h,w)

        action_candidates = torch.tensor(
            candidate_actions,
            dtype=torch.float32,
            device=device,
        )

        info = {
            "pixels": ctx[:, None].expand(B, S, H, *ctx.shape[2:]).contiguous(),
        }

        rollout_info = model.rollout(
            info,
            action_candidates,
            history_size=history_size,
        )

        pred = rollout_info["predicted_emb"]  # (B,S,T_pred,D)
        pred_terminal = pred[:, :, -1, :]     # (B,S,D)

        goal_emb = model.encode({"pixels": goal})["emb"][:, 0, :]  # (B,D)

        costs = ((pred_terminal - goal_emb[:, None, :]) ** 2).mean(dim=-1)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start
    memory = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    return costs.detach().cpu().numpy(), elapsed, memory


def rankdata_2d(x: np.ndarray) -> np.ndarray:
    """
    Convert values to ranks row-wise. Lower cost gets lower rank.
    Handles ties approximately by stable argsort order.
    """
    order = np.argsort(x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)

    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[1], dtype=np.float32)[None, :]

    return ranks


def spearman_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = rankdata_2d(a)
    rb = rankdata_2d(b)

    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)

    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1)) + 1e-12

    return num / den


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    top_a = np.argsort(a, axis=1)[:, :k]
    top_b = np.argsort(b, axis=1)[:, :k]

    overlaps = []
    for i in range(a.shape[0]):
        overlaps.append(len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / k)

    return np.asarray(overlaps, dtype=np.float32)


def evaluate_ranking(
    run_name: str,
    compressed_model_path: str,
    tag: str,
    *,
    num_contexts: int,
    num_candidates: int,
    rollout_blocks: int,
    noise_std: float,
    topk: int,
    device: str,
    seed: int,
) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obs = np.load(dirs["data"] / "obs.npy")
    actions = np.load(dirs["data"] / "actions.npy")
    terminateds = np.load(dirs["data"] / "terminateds.npy")
    truncateds = np.load(dirs["data"] / "truncateds.npy")

    action_mean, action_std = compute_action_block_stats(actions, cfg.action_block)

    examples = sample_contexts(
        obs,
        actions,
        terminateds,
        truncateds,
        history_size=cfg.history_size,
        action_block=cfg.action_block,
        rollout_blocks=rollout_blocks,
        num_contexts=num_contexts,
        seed=seed,
    )

    candidate_actions_raw = make_candidate_actions(
        examples["true_action_sequences"],
        num_candidates=num_candidates,
        noise_std=noise_std,
        seed=seed + 1,
    )

    candidate_actions = normalize_action_blocks(
        candidate_actions_raw,
        action_mean,
        action_std,
    )

    original = load_model(cfg.checkpoint_repo, device=device, model_path=None)
    compressed = load_model(cfg.checkpoint_repo, device=device, model_path=compressed_model_path)

    original_costs, original_time, original_mem = rollout_costs(
        original,
        context_pixels=examples["context_pixels"],
        goal_pixels=examples["goal_pixels"],
        candidate_actions=candidate_actions,
        history_size=cfg.history_size,
        device=device,
    )

    compressed_costs, compressed_time, compressed_mem = rollout_costs(
        compressed,
        context_pixels=examples["context_pixels"],
        goal_pixels=examples["goal_pixels"],
        candidate_actions=candidate_actions,
        history_size=cfg.history_size,
        device=device,
    )

    cost_diff = compressed_costs - original_costs
    spearman = spearman_per_row(original_costs, compressed_costs)
    top1 = (
        np.argmin(original_costs, axis=1)
        == np.argmin(compressed_costs, axis=1)
    ).astype(np.float32)
    overlap = topk_overlap(original_costs, compressed_costs, k=topk)

    out_dir = OUTPUT_ROOT / "phase2" / cfg.run_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranking_eval.json"

    results = {
        "run_name": cfg.run_name,
        "tag": tag,
        "compressed_model_path": compressed_model_path,
        "device": device,
        "num_contexts": int(original_costs.shape[0]),
        "num_candidates": int(original_costs.shape[1]),
        "history_size": int(cfg.history_size),
        "action_block": int(cfg.action_block),
        "rollout_blocks": int(rollout_blocks),
        "noise_std_raw_action_blocks": float(noise_std),
        "topk": int(topk),
        "cost_mse": float(np.mean(cost_diff**2)),
        "cost_mae": float(np.mean(np.abs(cost_diff))),
        "cost_bias": float(np.mean(cost_diff)),
        "spearman_mean": float(np.mean(spearman)),
        "spearman_std": float(np.std(spearman)),
        "top1_agreement": float(np.mean(top1)),
        "topk_overlap": float(np.mean(overlap)),
        "original_latency_sec": float(original_time),
        "compressed_latency_sec": float(compressed_time),
        "latency_ratio_compressed_over_original": float(compressed_time / original_time),
        "original_candidates_per_sec": float(original_costs.size / original_time),
        "compressed_candidates_per_sec": float(compressed_costs.size / compressed_time),
        "original_cuda_memory_bytes": original_mem,
        "compressed_cuda_memory_bytes": compressed_mem,
        "action_normalization": "local rollout mean/std diagnostic",
        "note": (
            "This is LeWM-specific action-ranking fidelity. It compares the original "
            "and compressed LeWM latent planning costs over the same candidate action "
            "sequences. This is closer to planning than transition MSE, but it is not "
            "yet full closed-loop environment success."
        ),
    }

    save_json(out_path, results)

    print("LeWM action-ranking fidelity:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-contexts", type=int, default=32)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--rollout-blocks", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_ranking(
        run_name=args.run,
        compressed_model_path=args.model_path,
        tag=args.tag,
        num_contexts=args.num_contexts,
        num_candidates=args.num_candidates,
        rollout_blocks=args.rollout_blocks,
        noise_std=args.noise_std,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
