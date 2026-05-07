from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import zstandard as zstd
from huggingface_hub import hf_hub_download


DATASETS = {
    "pusht": {
        "repo_id": "quentinll/lewm-pusht",
        "filename": "pusht_expert_train.h5.zst",
        "kind": "h5_zst",
        "expected": "pusht_expert_train.h5",
    },
    "tworoom": {
        "repo_id": "quentinll/lewm-tworooms",
        "filename": "tworoom.tar.zst",
        "kind": "tar_zst",
        "expected": "tworoom.h5",
    },
    "ogbench_cube": {
        "repo_id": "quentinll/lewm-cube",
        "filename": "cube_single_expert.tar.zst",
        "kind": "tar_zst",
        "expected": "ogbench/cube_single_expert.h5",
    },
}


def stablewm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")).expanduser()


def decompress_zst(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        print(f"exists: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"decompress: {src} -> {dst}")

    dctx = zstd.ZstdDecompressor()
    with src.open("rb") as fin, dst.open("wb") as fout:
        dctx.copy_stream(fin, fout)


def fetch_one(name: str, root: Path, overwrite: bool = False) -> None:
    spec = DATASETS[name]
    expected = root / spec["expected"]

    if expected.exists() and not overwrite:
        print(f"[{name}] already ready: {expected}")
        return

    print(f"\n[{name}] repo={spec['repo_id']} file={spec['filename']}")

    downloaded = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        repo_type="dataset",
        local_dir=root,
    )
    downloaded = Path(downloaded)

    if spec["kind"] == "h5_zst":
        out = root / spec["filename"].removesuffix(".zst")
        decompress_zst(downloaded, out, overwrite=overwrite)

    elif spec["kind"] == "tar_zst":
        tar_path = root / spec["filename"].removesuffix(".zst")
        decompress_zst(downloaded, tar_path, overwrite=overwrite)

        print(f"extract: {tar_path} -> {root}")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(root)

        # Normalize cube path to match LeWM eval config:
        # HDF5Dataset("ogbench/cube_single_expert") expects
        # $STABLEWM_HOME/ogbench/cube_single_expert.h5
        cube_root = root / "cube_single_expert.h5"
        cube_expected = root / "ogbench" / "cube_single_expert.h5"
        if name == "ogbench_cube" and cube_root.exists() and not cube_expected.exists():
            cube_expected.parent.mkdir(parents=True, exist_ok=True)
            cube_root.rename(cube_expected)

    else:
        raise ValueError(f"Unknown dataset kind: {spec['kind']}")

    if not expected.exists():
        matches = sorted(root.glob("**/*.h5"))
        print("HDF5 files found:")
        for m in matches:
            print(f"  {m.relative_to(root)}")
        raise FileNotFoundError(f"[{name}] expected {expected}, but it was not created.")

    print(f"[{name}] ready: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Datasets to fetch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-decompress/re-extract even if outputs exist.",
    )
    args = parser.parse_args()

    root = stablewm_home()
    root.mkdir(parents=True, exist_ok=True)

    print(f"STABLEWM_HOME={root}")

    for name in args.datasets:
        fetch_one(name, root=root, overwrite=args.overwrite)

    print("\nDone. HDF5 files:")
    for path in sorted(root.glob("**/*.h5")):
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
