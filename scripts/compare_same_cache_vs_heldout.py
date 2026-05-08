from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from oawc.compression.reports import save_json


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _heldout_base_tag(row: dict[str, str]) -> str:
    if row.get("base_tag"):
        return str(row["base_tag"])
    eval_tag = str(row.get("eval_tag") or row.get("tag") or "")
    return re.sub(r"_eval_s\d+_seed\d+$", "", eval_tag)


def _gap(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _md(rows: list[dict[str, Any]]) -> str:
    cols = [
        "base_tag",
        "same_cache_tag",
        "heldout_eval_tag",
        "same_cache_spearman",
        "heldout_spearman",
        "spearman_gap_heldout_minus_same",
        "same_cache_top5",
        "heldout_top5",
        "top5_gap_heldout_minus_same",
        "same_cache_regret",
        "heldout_regret",
        "regret_gap_heldout_minus_same",
    ]

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# Same-Cache vs Held-out Comparison (TwoRoom)",
        "",
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(c)) for c in cols) + "|")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--same-cache-csv",
        default="outputs/tables/compression_summary_tworoom.csv",
    )
    parser.add_argument(
        "--heldout-csv",
        default="outputs/tables/operator_split_summary_tworoom.csv",
    )
    args = parser.parse_args()

    same_path = Path(args.same_cache_csv)
    heldout_path = Path(args.heldout_csv)
    if not same_path.exists():
        raise FileNotFoundError(f"Missing same-cache summary: {same_path}")
    if not heldout_path.exists():
        raise FileNotFoundError(f"Missing held-out summary: {heldout_path}")

    same_rows = _read_csv(same_path)
    heldout_rows = _read_csv(heldout_path)

    same_by_tag = {str(r.get("tag", "")): r for r in same_rows}
    heldout_by_base = {}
    for row in heldout_rows:
        base = _heldout_base_tag(row)
        if base:
            heldout_by_base[base] = row

    keys = sorted(set(same_by_tag.keys()) | set(heldout_by_base.keys()))
    out_rows: list[dict[str, Any]] = []
    for base in keys:
        s = same_by_tag.get(base, {})
        h = heldout_by_base.get(base, {})
        same_spear = _to_float(s.get("spearman_mean"))
        hold_spear = _to_float(h.get("spearman_mean"))
        same_top5 = _to_float(s.get("top5_overlap_mean"))
        hold_top5 = _to_float(h.get("top5_overlap_mean"))
        same_regret = _to_float(s.get("teacher_regret_mean"))
        hold_regret = _to_float(h.get("teacher_regret_mean"))
        out_rows.append(
            {
                "base_tag": base,
                "same_cache_tag": s.get("tag"),
                "heldout_eval_tag": h.get("eval_tag") or h.get("tag"),
                "same_cache_spearman": same_spear,
                "heldout_spearman": hold_spear,
                "spearman_gap_heldout_minus_same": _gap(
                    same_spear,
                    hold_spear,
                ),
                "same_cache_top5": same_top5,
                "heldout_top5": hold_top5,
                "top5_gap_heldout_minus_same": _gap(same_top5, hold_top5),
                "same_cache_regret": same_regret,
                "heldout_regret": hold_regret,
                "regret_gap_heldout_minus_same": _gap(
                    same_regret,
                    hold_regret,
                ),
            }
        )

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "same_cache_vs_heldout_tworoom.csv"
    md_path = out_dir / "same_cache_vs_heldout_tworoom.md"
    json_path = out_dir / "same_cache_vs_heldout_tworoom.json"

    fields = [
        "base_tag",
        "same_cache_tag",
        "heldout_eval_tag",
        "same_cache_spearman",
        "heldout_spearman",
        "spearman_gap_heldout_minus_same",
        "same_cache_top5",
        "heldout_top5",
        "top5_gap_heldout_minus_same",
        "same_cache_regret",
        "heldout_regret",
        "regret_gap_heldout_minus_same",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    save_json(
        json_path,
        {
            "same_cache_csv": str(same_path),
            "heldout_csv": str(heldout_path),
            "rows": out_rows,
        },
    )
    md_path.write_text(_md(out_rows), encoding="utf-8")
    print("Same-cache vs held-out comparison written")
    print(f"  csv:  {csv_path}")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")


if __name__ == "__main__":
    main()
