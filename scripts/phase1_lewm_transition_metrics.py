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
    """
    T, E, A = actions.shape
    blocks = []

    for t in range(T - block_size + 1):
        block = actions[t : t + block_size]       # (B, E, A)
        block = np.transpose(block, (1, 0, 2))    # (E, B, A)
        block = block.reshape(E, block_size * A)
        blocks.append(block)

    return np.asarray(blocks, dtype=np.float32)


def build_pixel_transition_batches(
    obs: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    history_size: int,
    action_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds batches for the explicit LeWM transition model.

    obs: (T, E, H_img, W_img, C)
    actions: (T, E, A)

    Returns:
        pixel_windows: (N, H, H_img, W_img, C)
        action_windows: (N, H, action_block * A)
        target_pixels: (N, H_img, W_img, C)
        target_times: (N,)
        target_envs: (N,)
    """
    T, E = actions.shape[:2]
    done = np.logical_or(terminateds, truncateds)

    action_blocks = make_action_blocks(actions, action_block)

    pixel_windows = []
    action_windows = []
    target_pixels = []
    target_times = []
    target_envs = []

    max_start = T - history_size * action_block

    for t in range(max_start):
        frame_times = [t + k * action_block for k in range(history_size)]
        action_times = [t + k * action_block for k in range(history_size)]
        target_time = t + history_size * action_block

        for e in range(E):
            if done[t : target_time + 1, e].any():
                continue

            pixel_windows.append(obs[frame_times, e])
            action_windows.append(action_blocks[action_times, e])
            target_pixels.append(obs[target_time, e])
            target_times.append(target_time)
            target_envs.append(e)

    return (
        np.asarray(pixel_windows, dtype=np.uint8),
        np.asarray(action_windows, dtype=np.float32),
        np.asarray(target_pixels, dtype=np.uint8),
        np.asarray(target_times, dtype=np.int64),
        np.asarray(target_envs, dtype=np.int64),
    )


def to_pixel_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    """
    x: (B, H, H_img, W_img, C)
    returns: (B, H, C, H_img, W_img)
    """
    t = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    return t.permute(0, 1, 4, 2, 3).contiguous()


def to_action_tensor(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32, device=device)


def lewm_transition_metrics(
    run_name: str,
    batch_size: int = 32,
    device: str = "auto",
    max_batches: int | None = None,
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

    pixel_windows, action_windows, target_pixels, target_times, target_envs = (
        build_pixel_transition_batches(
            obs=obs,
            actions=actions,
            terminateds=terminateds,
            truncateds=truncateds,
            history_size=cfg.history_size,
            action_block=cfg.action_block,
        )
    )

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)

    num_examples = len(pixel_windows)
    if num_examples == 0:
        raise RuntimeError("No valid transition examples were built.")

    sq_errors = []
    last_latent_sq_errors = []
    pred_norms = []
    target_norms = []

    start = time.time()

    with torch.no_grad():
        num_batches = 0

        for i in tqdm(range(0, num_examples, batch_size), desc="LeWM transition"):
            if max_batches is not None and num_batches >= max_batches:
                break

            pw = pixel_windows[i : i + batch_size]
            aw = action_windows[i : i + batch_size]
            tp = target_pixels[i : i + batch_size]

            x_hist = to_pixel_tensor(pw, device)
            a_hist = to_action_tensor(aw, device)

            x_target = torch.tensor(tp, dtype=torch.float32, device=device) / 255.0
            x_target = x_target.permute(0, 3, 1, 2).unsqueeze(1).contiguous()

            hist_batch = {
                "pixels": x_hist,
                "action": a_hist,
            }
            target_batch = {
                "pixels": x_target,
            }

            hist_out = model.encode(hist_batch)
            target_out = model.encode(target_batch)

            emb = hist_out["emb"]
            act_emb = hist_out["act_emb"]
            target = target_out["emb"]

            if target.ndim == 3:
                target = target.squeeze(1)

            pred = model.predict(emb, act_emb)

            # LeWM predict may return either the full predicted sequence or the final next embedding.
            if pred.ndim == 3:
                pred_final = pred[:, -1]
            else:
                pred_final = pred

            last_latent = emb[:, -1]

            sq_error = ((pred_final - target) ** 2).mean(dim=1)
            last_error = ((last_latent - target) ** 2).mean(dim=1)

            sq_errors.append(sq_error.detach().cpu().numpy())
            last_latent_sq_errors.append(last_error.detach().cpu().numpy())
            pred_norms.append(torch.linalg.norm(pred_final, dim=1).detach().cpu().numpy())
            target_norms.append(torch.linalg.norm(target, dim=1).detach().cpu().numpy())

            num_batches += 1

    elapsed = time.time() - start

    sq_errors_all = np.concatenate(sq_errors)
    last_errors_all = np.concatenate(last_latent_sq_errors)
    pred_norms_all = np.concatenate(pred_norms)
    target_norms_all = np.concatenate(target_norms)

    results = {
        "run_name": cfg.run_name,
        "model_family": "LeWM",
        "checkpoint_repo": cfg.checkpoint_repo,
        "num_examples_total": int(num_examples),
        "num_examples_eval": int(len(sq_errors_all)),
        "history_size": int(cfg.history_size),
        "action_block": int(cfg.action_block),
        "batch_size": int(batch_size),
        "device": device,
        "transition_mse": float(sq_errors_all.mean()),
        "transition_mse_std": float(sq_errors_all.std()),
        "last_latent_baseline_mse": float(last_errors_all.mean()),
        "last_latent_baseline_mse_std": float(last_errors_all.std()),
        "pred_norm_mean": float(pred_norms_all.mean()),
        "target_norm_mean": float(target_norms_all.mean()),
        "elapsed_sec": float(elapsed),
        "note": (
            "This evaluates the explicit LeWM transition predictor. "
            "It is LeWM-specific, not yet a general WorldModel interface."
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
    args = parser.parse_args()

    lewm_transition_metrics(
        run_name=args.run,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()