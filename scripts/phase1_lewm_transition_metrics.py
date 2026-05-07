from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from oawc.config import get_run_config
from oawc.models.lewm_loader import load_lewm_from_hf
from oawc.paths import phase1_dirs, save_json


def make_action_blocks(actions: np.ndarray, block_size: int) -> np.ndarray:
    """
    actions: (T, E, A)
    returns: (T - block_size + 1, E, block_size * A)

    This matches LeWM's frameskip/action-block idea:
    consecutive raw environment actions are concatenated into one action block.
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape (T, E, A), got {actions.shape}")

    T, E, A = actions.shape

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    if block_size > T:
        raise ValueError(f"block_size={block_size} exceeds T={T}")

    blocks = []

    for t in range(T - block_size + 1):
        block = actions[t : t + block_size]       # (B, E, A)
        block = np.transpose(block, (1, 0, 2))    # (E, B, A)
        block = block.reshape(E, block_size * A)
        blocks.append(block)

    return np.asarray(blocks, dtype=np.float32)


def compute_action_block_stats(actions: np.ndarray, action_block: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagnostic action normalization.

    LeWM training normalizes non-pixel columns using dataset-wide mean/std.
    Here we estimate mean/std from our collected rollout only. This is not the
    final paper-correct version, but it tests whether preprocessing mismatch is
    causing bad transition metrics.
    """
    action_blocks = make_action_blocks(actions, action_block)  # (T-B+1, E, B*A)
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


def build_lewm_training_sequences(
    obs: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    history_size: int,
    action_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the LeWM-style prediction contract.

    pixel_seq:    (N, H+1, H_img, W_img, C)
    action_seq:   (N, H, action_block * action_dim)
    target_times: (N,)
    target_envs:  (N,)

    The model receives pixel_seq[:, :-1] and action_seq,
    and predicts embeddings corresponding to pixel_seq[:, 1:].
    """
    if obs.ndim != 5:
        raise ValueError(f"Expected obs shape (T, E, H, W, C), got {obs.shape}")

    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape (T, E, A), got {actions.shape}")

    T, E = actions.shape[:2]

    if obs.shape[0] != T or obs.shape[1] != E:
        raise ValueError(
            f"Obs/actions mismatch: obs={obs.shape}, actions={actions.shape}"
        )

    done = np.logical_or(terminateds, truncateds)

    if done.shape != (T, E):
        raise ValueError(f"Expected done shape {(T, E)}, got {done.shape}")

    action_blocks = make_action_blocks(actions, action_block)

    pixel_seq = []
    action_seq = []
    target_times = []
    target_envs = []

    max_start = T - history_size * action_block

    for t in range(max_start):
        frame_times = [t + k * action_block for k in range(history_size + 1)]
        action_times = [t + k * action_block for k in range(history_size)]
        final_target_time = t + history_size * action_block

        for e in range(E):
            if done[t : final_target_time + 1, e].any():
                continue

            pixel_seq.append(obs[frame_times, e])
            action_seq.append(action_blocks[action_times, e])
            target_times.append(final_target_time)
            target_envs.append(e)

    return (
        np.asarray(pixel_seq, dtype=np.uint8),
        np.asarray(action_seq, dtype=np.float32),
        np.asarray(target_times, dtype=np.int64),
        np.asarray(target_envs, dtype=np.int64),
    )


def to_pixel_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    """
    x: (B, T, H, W, C), uint8
    returns: (B, T, C, H, W), ImageNet-normalized

    LeWM training uses stable-pretraining's ToImage with ImageNet stats,
    so this is closer than raw x / 255.
    """
    t = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    t = t.permute(0, 1, 4, 2, 3).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    return (t - mean) / std


def to_action_tensor(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32, device=device)


def lewm_transition_metrics(
    run_name: str,
    batch_size: int = 32,
    device: str = "auto",
    max_batches: int | None = None,
    normalize_actions: bool = True,
) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    obs_path = dirs["data"] / "obs.npy"
    actions_path = dirs["data"] / "actions.npy"
    terminateds_path = dirs["data"] / "terminateds.npy"
    truncateds_path = dirs["data"] / "truncateds.npy"
    out_path = dirs["results"] / "lewm_transition_metrics.json"

    for path in [obs_path, actions_path, terminateds_path, truncateds_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obs = np.load(obs_path)
    actions = np.load(actions_path)
    terminateds = np.load(terminateds_path)
    truncateds = np.load(truncateds_path)

    action_mean, action_std = compute_action_block_stats(
        actions=actions,
        action_block=cfg.action_block,
    )

    pixel_seq, action_seq, target_times, target_envs = build_lewm_training_sequences(
        obs=obs,
        actions=actions,
        terminateds=terminateds,
        truncateds=truncateds,
        history_size=cfg.history_size,
        action_block=cfg.action_block,
    )

    if len(pixel_seq) == 0:
        raise RuntimeError("No valid LeWM transition sequences were built.")

    if normalize_actions:
        action_seq = normalize_action_blocks(action_seq, action_mean, action_std)

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)

    seq_mse_values = []
    final_mse_values = []
    per_step_sums = None
    per_step_counts = 0

    copy_seq_mse_values = []
    copy_final_mse_values = []

    pred_norms = []
    target_norms = []

    start = time.time()
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(pixel_seq), batch_size), desc="LeWM transition"):
            if max_batches is not None and num_batches >= max_batches:
                break

            px = pixel_seq[i : i + batch_size]
            ac = action_seq[i : i + batch_size]

            pixels = to_pixel_tensor(px, device)
            actions_t = to_action_tensor(ac, device)

            batch = {
                "pixels": pixels,
                "action": actions_t,
            }

            encoded = model.encode(batch)

            emb_all = encoded["emb"]        # (B, H+1, D)
            act_emb = encoded["act_emb"]    # (B, H, D)

            emb_context = emb_all[:, :-1]   # (B, H, D)
            emb_target = emb_all[:, 1:]     # (B, H, D)

            pred = model.predict(emb_context, act_emb)  # (B, H, D)

            if pred.shape != emb_target.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: pred={pred.shape}, "
                    f"target={emb_target.shape}"
                )

            per_ex_seq_mse = ((pred - emb_target) ** 2).mean(dim=(1, 2))
            per_ex_final_mse = ((pred[:, -1] - emb_target[:, -1]) ** 2).mean(dim=1)

            # Copy baseline: predict z_{k+1} := z_k.
            copy_pred = emb_context
            copy_seq_mse = ((copy_pred - emb_target) ** 2).mean(dim=(1, 2))
            copy_final_mse = ((copy_pred[:, -1] - emb_target[:, -1]) ** 2).mean(dim=1)

            per_step = ((pred - emb_target) ** 2).mean(dim=2).sum(dim=0)  # (H,)
            if per_step_sums is None:
                per_step_sums = per_step.detach().cpu()
            else:
                per_step_sums += per_step.detach().cpu()

            per_step_counts += pred.shape[0]

            seq_mse_values.append(per_ex_seq_mse.detach().cpu().numpy())
            final_mse_values.append(per_ex_final_mse.detach().cpu().numpy())
            copy_seq_mse_values.append(copy_seq_mse.detach().cpu().numpy())
            copy_final_mse_values.append(copy_final_mse.detach().cpu().numpy())

            pred_norms.append(torch.linalg.norm(pred[:, -1], dim=1).detach().cpu().numpy())
            target_norms.append(torch.linalg.norm(emb_target[:, -1], dim=1).detach().cpu().numpy())

            num_batches += 1

    elapsed = time.time() - start

    seq_mse = np.concatenate(seq_mse_values)
    final_mse = np.concatenate(final_mse_values)
    copy_seq_mse = np.concatenate(copy_seq_mse_values)
    copy_final_mse = np.concatenate(copy_final_mse_values)
    pred_norms_all = np.concatenate(pred_norms)
    target_norms_all = np.concatenate(target_norms)

    per_step_mse = (per_step_sums / per_step_counts).numpy().tolist()

    results = {
        "run_name": cfg.run_name,
        "model_family": "LeWM",
        "checkpoint_repo": cfg.checkpoint_repo,
        "num_examples_total": int(len(pixel_seq)),
        "num_examples_eval": int(len(seq_mse)),
        "history_size": int(cfg.history_size),
        "action_block": int(cfg.action_block),
        "batch_size": int(batch_size),
        "device": device,
        "pixel_preprocessing": "ImageNet normalization",
        "action_normalization": (
            "local rollout mean/std diagnostic"
            if normalize_actions
            else "none/raw action blocks"
        ),
        "action_mean": action_mean.squeeze(0).tolist(),
        "action_std": action_std.squeeze(0).tolist(),
        "sequence_transition_mse": float(seq_mse.mean()),
        "sequence_transition_mse_std": float(seq_mse.std()),
        "final_transition_mse": float(final_mse.mean()),
        "final_transition_mse_std": float(final_mse.std()),
        "copy_sequence_baseline_mse": float(copy_seq_mse.mean()),
        "copy_final_baseline_mse": float(copy_final_mse.mean()),
        "per_step_transition_mse": per_step_mse,
        "pred_final_norm_mean": float(pred_norms_all.mean()),
        "target_final_norm_mean": float(target_norms_all.mean()),
        "elapsed_sec": float(elapsed),
        "note": (
            "This evaluates the explicit LeWM transition predictor using the LeWM "
            "training contract: encode H+1 frames, predict emb[:, 1:] from "
            "emb[:, :-1] and H action embeddings. This is LeWM-specific. "
            "Action normalization currently uses local rollout statistics as a "
            "diagnostic; the final version should use the original training dataset stats."
        ),
    }

    save_json(out_path, results)

    print("LeWM transition metrics:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--no-action-normalization",
        action="store_true",
        help="Use raw action blocks instead of local mean/std normalized blocks.",
    )
    args = parser.parse_args()

    lewm_transition_metrics(
        run_name=args.run,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
        normalize_actions=not args.no_action_normalization,
    )


if __name__ == "__main__":
    main()
