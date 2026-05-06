from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from oawc.config import get_run_config
from oawc.models.lewm_loader import load_lewm_from_hf
from oawc.paths import phase1_dirs, save_json


def encode_lewm(
    run_name: str,
    batch_size: int = 64,
    device: str = "auto",
    overwrite: bool = False,
) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    obs_path = dirs["data"] / "obs.npy"
    z_path = dirs["latents"] / "lewm_z.npy"
    metadata_path = dirs["latents"] / "lewm_metadata.json"

    if z_path.exists() and not overwrite:
        z = np.load(z_path)
        print(f"LeWM latents already exist: {z_path}")
        print(f"shape: {z.shape}, dtype: {z.dtype}")
        return

    if not obs_path.exists():
        raise FileNotFoundError(f"Missing observations: {obs_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obs = np.load(obs_path)
    if obs.ndim != 5:
        raise ValueError(f"Expected obs shape (T, E, H, W, C), got {obs.shape}")

    T, E, H, W, C = obs.shape
    obs_flat = obs.reshape(T * E, H, W, C)

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)

    zs: list[np.ndarray] = []
    start = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(obs_flat), batch_size), desc=f"Encoding {run_name}"):
            batch = obs_flat[i : i + batch_size]

            x = torch.tensor(batch, dtype=torch.float32, device=device) / 255.0
            x = x.permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, C, H, W)

            out = model.encode({"pixels": x})
            z = out["emb"]

            if z.ndim == 3:
                z = z.squeeze(1)

            zs.append(z.detach().cpu().float().numpy())

    z_all = np.concatenate(zs, axis=0).astype(np.float32)
    elapsed = time.time() - start

    np.save(z_path, z_all)

    save_json(
        metadata_path,
        {
            "run_name": cfg.run_name,
            "checkpoint_repo": cfg.checkpoint_repo,
            "obs_path": obs_path,
            "latent_path": z_path,
            "obs_shape": list(obs.shape),
            "obs_flat_shape": list(obs_flat.shape),
            "latent_shape": list(z_all.shape),
            "latent_dtype": str(z_all.dtype),
            "device": device,
            "batch_size": batch_size,
            "elapsed_sec": elapsed,
        },
    )

    print("Saved LeWM latents:")
    print(f"  path: {z_path}")
    print(f"  shape: {z_all.shape}")
    print(f"  dtype: {z_all.dtype}")
    print(f"  elapsed_sec: {elapsed:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    encode_lewm(
        run_name=args.run,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()