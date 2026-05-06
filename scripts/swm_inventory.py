from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    print("Python:", sys.version)
    print("STABLEWM_HOME:", os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel"))

    try:
        import stable_worldmodel as swm
        print("stable_worldmodel import: OK")
        print("stable_worldmodel module:", swm)
    except Exception as exc:
        print("stable_worldmodel import: FAILED")
        raise exc

    run(["swm", "envs"])
    run(["swm", "datasets"])
    run(["swm", "checkpoints"])


if __name__ == "__main__":
    main()