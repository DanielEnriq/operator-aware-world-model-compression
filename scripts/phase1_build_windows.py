from __future__ import annotations

import argparse
import time

import numpy as np

from oawc.config import get_run_config
from oawc.paths import phase1_dirs, save_json


def make_action_blocks(actions: np.ndarray, block_size: int) -> np.ndarray:
    """
    actions: (T, E, A)
    returns: (T - block_size + 1, E, block_size * A)
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
        block = actions[t : t + block_size]      # (B, E, A)
        block = np.transpose(block, (1, 0, 2))   # (E, B, A)
        block = block.reshape(E, block_size * A)
        blocks.append(block)

    return np.asarray(blocks, dtype=np.float32)


def build_transition_windows(
    run_name: str,
    latent_name: str = "lewm",
    overwrite: bool = False,
) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    z_path = dirs["latents"] / f"{latent_name}_z.npy"
    actions_path = dirs["data"] / "actions.npy"
    terminateds_path = dirs["data"] / "terminateds.npy"
    truncateds_path = dirs["data"] / "truncateds.npy"

    out_path = dirs["windows"] / f"{latent_name}_transition_windows.npz"
    metadata_path = dirs["windows"] / f"{latent_name}_transition_windows_metadata.json"

    if out_path.exists() and not overwrite:
        data = np.load(out_path)
        print(f"Transition windows already exist: {out_path}")
        print(f"emb_windows: {data['emb_windows'].shape}")
        print(f"act_windows: {data['act_windows'].shape}")
        print(f"targets:     {data['targets'].shape}")
        return

    for path in [z_path, actions_path, terminateds_path, truncateds_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    z_flat = np.load(z_path)
    actions = np.load(actions_path)
    terminateds = np.load(terminateds_path)
    truncateds = np.load(truncateds_path)

    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape (T, E, A), got {actions.shape}")

    T, E, A = actions.shape

    if z_flat.ndim != 2:
        raise ValueError(
            f"This window builder currently expects vector latents of shape (T*E, D), "
            f"got {z_flat.shape}"
        )

    if z_flat.shape[0] != T * E:
        raise ValueError(
            f"Latent count mismatch: z_flat.shape[0]={z_flat.shape[0]}, "
            f"but T*E={T * E} from actions."
        )

    D = z_flat.shape[1]
    z_seq = z_flat.reshape(T, E, D)

    done = np.logical_or(terminateds, truncateds)
    if done.shape != (T, E):
        raise ValueError(f"Expected done shape {(T, E)}, got {done.shape}")

    H = cfg.history_size
    B = cfg.action_block

    action_blocks = make_action_blocks(actions, B)

    emb_windows = []
    act_windows = []
    targets = []
    target_times = []
    target_envs = []

    max_start = T - H * B

    for t in range(max_start):
        frame_times = [t + k * B for k in range(H)]
        action_times = [t + k * B for k in range(H)]
        target_time = t + H * B

        for e in range(E):
            if done[t : target_time + 1, e].any():
                continue

            emb_windows.append(z_seq[frame_times, e])
            act_windows.append(action_blocks[action_times, e])
            targets.append(z_seq[target_time, e])
            target_times.append(target_time)
            target_envs.append(e)

    emb_windows = np.asarray(emb_windows, dtype=np.float32)
    act_windows = np.asarray(act_windows, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.int64)
    target_envs = np.asarray(target_envs, dtype=np.int64)

    np.savez_compressed(
        out_path,
        emb_windows=emb_windows,
        act_windows=act_windows,
        targets=targets,
        target_times=target_times,
        target_envs=target_envs,
    )

    save_json(
        metadata_path,
        {
            "run_name": cfg.run_name,
            "latent_name": latent_name,
            "latent_path": z_path,
            "actions_path": actions_path,
            "output_path": out_path,
            "T": T,
            "E": E,
            "action_dim": A,
            "latent_dim": D,
            "history_size": H,
            "action_block": B,
            "num_windows": int(len(targets)),
            "emb_windows_shape": list(emb_windows.shape),
            "act_windows_shape": list(act_windows.shape),
            "targets_shape": list(targets.shape),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": (
                "This builder is environment-general for vector latents, "
                "but not yet general for patch-token latent tensors."
            ),
        },
    )

    print("Saved transition windows:")
    print(f"  path:        {out_path}")
    print(f"  emb_windows: {emb_windows.shape}")
    print(f"  act_windows: {act_windows.shape}")
    print(f"  targets:     {targets.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--latent-name", default="lewm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_transition_windows(
        run_name=args.run,
        latent_name=args.latent_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()