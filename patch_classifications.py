#!/usr/bin/env python3
"""
Apply YAML-declared post-classification patches to a classified parquet.

Patches live in patches.yaml. Each patch either targets a CN allowlist or
matches rows via regex over the Education + QualificationSummary text.
Flipped rows get an `edu_patch_reason` column so the override is visible
downstream.

Called by prep_site_data.py automatically. Can also be run standalone to
inspect what a patches.yaml change would do:

    python patch_classifications.py --input data/2210_classified.parquet --dry-run
    python patch_classifications.py --input data/2210_classified.parquet --apply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
DEFAULT_PATCHES = ROOT / "patches.yaml"
DEFAULT_INPUT = ROOT / "data" / "2210_classified.parquet"

TEXT_COLUMNS = ("education", "qualification_summary")


def _combined_text(row: pd.Series) -> str:
    return "\n\n".join(str(row.get(c, "") or "") for c in TEXT_COLUMNS)


def _compile_list(patterns: list[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in (patterns or [])]


def _load_patches(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("patches", []) or []


def _rows_matching(df: pd.DataFrame, patch: dict) -> pd.Index:
    match = patch.get("match", {}) or {}
    mask = pd.Series(True, index=df.index)

    cns = match.get("control_numbers")
    if cns:
        cn_set = {str(c) for c in cns}
        mask &= df["usajobs_control_number"].astype(str).isin(cn_set)

    cur_cat = match.get("current_category")
    if cur_cat:
        mask &= df["edu_category"] == cur_cat

    text_any = _compile_list(match.get("text_any"))
    text_none = _compile_list(match.get("text_none"))
    if text_any or text_none:
        # Only compute text for rows still in the running.
        sub = df.loc[mask]
        if len(sub):
            text = sub.apply(_combined_text, axis=1)
            if text_any:
                any_mask = text.apply(lambda t: any(r.search(t) for r in text_any))
                mask.loc[sub.index] &= any_mask
            if text_none:
                sub = df.loc[mask]  # refresh after text_any narrowing
                if len(sub):
                    text2 = sub.apply(_combined_text, axis=1)
                    none_mask = text2.apply(lambda t: not any(r.search(t) for r in text_none))
                    mask.loc[sub.index] &= none_mask

    return df.index[mask]


def apply_patches(df: pd.DataFrame, patches_path: Path = DEFAULT_PATCHES) -> pd.DataFrame:
    """Return a copy of df with patches applied. Safe to call repeatedly."""
    patches = _load_patches(patches_path)
    if not patches:
        return df.copy()

    out = df.copy()
    if "edu_patch_reason" not in out.columns:
        out["edu_patch_reason"] = ""

    for patch in patches:
        name = patch.get("name", "<unnamed>")
        idx = _rows_matching(out, patch)
        if len(idx) == 0:
            continue
        new_fields = patch.get("set", {}) or {}
        for field, val in new_fields.items():
            out.loc[idx, field] = val
        out.loc[idx, "edu_patch_reason"] = name

    return out


def _summarize(before: pd.DataFrame, after: pd.DataFrame) -> None:
    changed = before["edu_category"].values != after["edu_category"].values
    n_changed = int(changed.sum())
    print(f"Patched rows: {n_changed} / {len(before):,}")
    if n_changed == 0:
        return
    summary = pd.DataFrame({
        "from": before.loc[changed, "edu_category"].values,
        "to": after.loc[changed, "edu_category"].values,
        "patch": after.loc[changed, "edu_patch_reason"].values,
    })
    print("\nTransitions:")
    print(summary.groupby(["patch", "from", "to"]).size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Classified parquet to patch")
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES,
                        help="YAML file declaring patch rules")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the patched parquet "
                             "(default: <input> with .patched.parquet suffix). "
                             "Ignored under --dry-run.")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="Report what would change (default)")
    g.add_argument("--apply", dest="dry_run", action="store_false",
                   help="Actually write the output parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    patched = apply_patches(df, args.patches)
    _summarize(df, patched)

    if args.dry_run:
        print("\n(dry run — re-run with --apply to write)")
        return

    out = args.output or args.input.with_suffix(".patched.parquet")
    patched.to_parquet(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
