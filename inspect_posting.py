#!/usr/bin/env python3
"""
Look up a USAJobs control number across the classified parquets and
print everything you'd want to see before deciding a patch.

Usage:
    python inspect.py 708210200
    python inspect.py 708210200 --full   # print full education/qual text (default truncates)
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
DATASETS = [
    ("2210", ROOT / "data" / "2210_classified.parquet", ROOT / "patches.yaml"),
    ("all_series", ROOT / "data" / "all_series_classified.parquet", ROOT / "patches_all_series.yaml"),
]


def find_row(cn: str) -> list[tuple[str, pd.Series, Path, Path]]:
    """Return list of (dataset_name, row, parquet_path, patches_path)."""
    hits = []
    for name, parquet, patches in DATASETS:
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        match = df[df["usajobs_control_number"].astype(str) == cn]
        if len(match):
            hits.append((name, match.iloc[0], parquet, patches))
    return hits


def find_in_patches(cn: str, patches_path: Path) -> list[str]:
    """Return names of patches that currently list this CN."""
    if not patches_path.exists():
        return []
    raw = yaml.safe_load(patches_path.read_text()) or {}
    out = []
    for patch in (raw.get("patches") or []):
        cns = ((patch.get("match") or {}).get("control_numbers")) or []
        if str(cn) in {str(c) for c in cns}:
            out.append(patch.get("name", "<unnamed>"))
    return out


def dump_text(label: str, text: str, full: bool) -> None:
    text = text or ""
    shown = text if full else (text[:800] + ("..." if len(text) > 800 else ""))
    print(f"\n--- {label} ({len(text)} chars) ---")
    print(shown.strip() or "(empty)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cn", help="USAJobs control number")
    parser.add_argument("--full", action="store_true",
                        help="Print full education/qualification text without truncation")
    args = parser.parse_args()

    hits = find_row(args.cn)
    if not hits:
        print(f"CN {args.cn} not found in any classified parquet.")
        return

    for name, row, parquet_path, patches_path in hits:
        print(f"\n{'=' * 78}")
        print(f"Dataset: {name}  |  parquet: {parquet_path}")
        print(f"Patches file: {patches_path.name}")
        print(f"{'=' * 78}")

        fields = [
            ("Title", row.get("position_title")),
            ("Agency", row.get("agency")),
            ("Department", row.get("department")),
            ("URL", row.get("usajobs_url")),
            ("Grade", f"{row.get('min_grade')}-{row.get('max_grade')}"),
            ("Source year", row.get("source_year")),
            ("Data source", row.get("data_source")),
        ]
        if name == "all_series":
            fields.insert(2, ("Series", f"{row.get('series_num')} {row.get('series_title', '')}"))
            fields.insert(3, ("OPM tier", f"{row.get('opm_tier')}"
                              + (f" / {row.get('opm_mandatory_type')}" if row.get("opm_mandatory_type") else "")))
        for k, v in fields:
            print(f"  {k:12s}: {v}")

        print("\nClassification:")
        print(f"  edu_category      : {row.get('edu_category')}")
        print(f"  edu_patch_reason  : {row.get('edu_patch_reason') or '(none)'}")
        print(f"  edu_key_quote     : {str(row.get('edu_key_quote', ''))[:400]}")
        print(f"  edu_reasoning     : {str(row.get('edu_reasoning', ''))[:400]}")

        patch_hits = find_in_patches(args.cn, patches_path)
        if patch_hits:
            print(f"\nListed in patches under: {', '.join(patch_hits)}")

        dump_text("Education", str(row.get("education", "")), args.full)
        dump_text("Qualifications", str(row.get("qualification_summary", "")), args.full)


if __name__ == "__main__":
    main()
