from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
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


def make_action_blocks(actions: np.ndarray, block_size: int) -> np.ndarray:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape (T,E,A), got {actions.shape}")

    T, E, A = actions.shape
    blocks = []

    for t in range(T - block_size + 1):
        block = actions[t : t + block_size]       # (B,E,A)
        block = np.transpose(block, (1, 0, 2))    # (E,B,A)
        block = block.reshape(E, block_size * A)
        blocks.append(block)

    return np.asarray(blocks, dtype=np.float32)


def compute_action_block_stats(actions: np.ndarray, action_block: int):
    action_blocks = make_action_blocks(actions, action_block)
    flat = action_blocks.reshape(-1, action_blocks.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)
    return mean, std


def normalize_action_blocks(action_blocks: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((action_blocks - mean) / (std + 1e-6)).astype(np.float32)


def build_sequences(
    obs: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    *,
    history_size: int,
    action_block: int,
):
    """
    pixel_seq:  (N, H+1, h, w, c)
    action_seq: (N, H, action_block * action_dim)
    """
    T, E = actions.shape[:2]
    done = np.logical_or(terminateds, truncateds)
    action_blocks = make_action_blocks(actions, action_block)

    pixel_seq = []
    action_seq = []

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

    return (
        np.asarray(pixel_seq, dtype=np.uint8),
        np.asarray(action_seq, dtype=np.float32),
    )


def to_pixel_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    t = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    t = t.permute(0, 1, 4, 2, 3).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    return (t - mean) / std


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_predictor(model: nn.Module) -> None:
    for p in model.predictor.parameters():
        p.requires_grad_(True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_distill(
    run_name: str,
    compressed_model_path: str,
    tag: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    beta_target: float,
    device: str,
    seed: int,
) -> None:
    ensure_lewm_source_on_path()

    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs = np.load(dirs["data"] / "obs.npy")
    actions = np.load(dirs["data"] / "actions.npy")
    terminateds = np.load(dirs["data"] / "terminateds.npy")
    truncateds = np.load(dirs["data"] / "truncateds.npy")

    action_mean, action_std = compute_action_block_stats(actions, cfg.action_block)

    pixel_seq, action_seq = build_sequences(
        obs,
        actions,
        terminateds,
        truncateds,
        history_size=cfg.history_size,
        action_block=cfg.action_block,
    )

    action_seq = normalize_action_blocks(action_seq, action_mean, action_std)

    if len(pixel_seq) == 0:
        raise RuntimeError("No valid training sequences were built.")

    teacher = load_lewm_from_hf(cfg.checkpoint_repo, device=device)
    teacher.eval()
    freeze_all(teacher)

    student = torch.load(compressed_model_path, map_location=device, weights_only=False)
    student.to(device)
    student.eval()

    freeze_all(student)
    unfreeze_predictor(student)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    n = len(pixel_seq)
    rng = np.random.default_rng(seed)

    out_dir = OUTPUT_ROOT / "phase3" / cfg.run_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    start = time.time()

    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_losses = []

        pbar = tqdm(range(0, n, batch_size), desc=f"distill epoch {epoch + 1}/{epochs}")

        for start_idx in pbar:
            idx = order[start_idx : start_idx + batch_size]

            px = to_pixel_tensor(pixel_seq[idx], device)
            ac = torch.tensor(action_seq[idx], dtype=torch.float32, device=device)

            batch = {"pixels": px, "action": ac}

            with torch.no_grad():
                teacher_out = teacher.encode(batch)
                teacher_emb_all = teacher_out["emb"]
                teacher_act_emb = teacher_out["act_emb"]

                emb_context = teacher_emb_all[:, :-1]
                emb_target = teacher_emb_all[:, 1:]

                teacher_pred = teacher.predict(emb_context, teacher_act_emb)

            # Use teacher embeddings/actions as calibration inputs so the student
            # is directly trained to preserve the deployed transition operator.
            student_pred = student.predict(emb_context.detach(), teacher_act_emb.detach())

            loss_teacher = torch.mean((student_pred - teacher_pred.detach()) ** 2)
            loss_target = torch.mean((student_pred - emb_target.detach()) ** 2)
            loss = loss_teacher + beta_target * loss_target

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            value = float(loss.detach().cpu())
            epoch_losses.append(value)
            pbar.set_postfix(loss=value)

        losses.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(epoch_losses)),
            }
        )

    elapsed = time.time() - start

    model_path = out_dir / "model.pt"
    metadata_path = out_dir / "metadata.json"

    student.eval()
    torch.save(student, model_path)

    metadata = {
        "run_name": cfg.run_name,
        "tag": tag,
        "method": "predictor_operator_distillation",
        "base_compressed_model_path": compressed_model_path,
        "output_model_path": model_path,
        "checkpoint_repo": cfg.checkpoint_repo,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "beta_target": beta_target,
        "device": device,
        "num_sequences": int(n),
        "trainable_params": int(count_trainable_params(student)),
        "elapsed_sec": float(elapsed),
        "losses": losses,
        "objective": "MSE(student_pred, teacher_pred) + beta_target * MSE(student_pred, target_emb)",
        "note": (
            "This fine-tunes only the compressed predictor. Encoder, projector, "
            "action_encoder, and pred_proj remain frozen. This is transition-operator "
            "distillation, not yet planner-ranking-aware fine-tuning."
        ),
    }

    save_json(metadata_path, metadata)

    print("Saved operator-distilled compressed model:")
    print(f"  model: {model_path}")
    print(f"  meta:  {metadata_path}")
    print(f"  trainable params: {count_trainable_params(student):,}")
    print(f"  elapsed_sec: {elapsed:.2f}")
    print(f"  final loss: {losses[-1]['loss']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-target", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_distill(
        run_name=args.run,
        compressed_model_path=args.model_path,
        tag=args.tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_target=args.beta_target,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
