from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np

from oawc.config import get_run_config
from oawc.paths import phase1_dirs


def visualize_rollout(run_name: str, env_idx: int = 0, fps: int = 15) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    obs_path = dirs["data"] / "obs.npy"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing rollout observations: {obs_path}")

    obs = np.load(obs_path)

    if obs.ndim != 5:
        raise ValueError(f"Expected obs shape (T, E, H, W, C), got {obs.shape}")

    T, E = obs.shape[:2]
    if not (0 <= env_idx < E):
        raise ValueError(f"env_idx={env_idx} out of range for E={E}")

    frame_indices = sorted(set([0, T // 4, T // 2, (3 * T) // 4, T - 1]))

    for t in frame_indices:
        out_path = dirs["figures"] / f"env{env_idx}_t{t:03d}.png"
        imageio.imwrite(out_path, obs[t, env_idx])
        print(f"Saved still: {out_path}")

    video_path = dirs["videos"] / f"env{env_idx}_rollout.mp4"
    frames = [obs[t, env_idx] for t in range(T)]
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved video: {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    visualize_rollout(args.run, env_idx=args.env_idx, fps=args.fps)


if __name__ == "__main__":
    main()