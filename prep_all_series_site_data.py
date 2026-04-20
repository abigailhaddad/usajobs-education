#!/usr/bin/env python3
"""Build site/all_series_data.json from data/all_series_classified.parquet.

Applies patches_all_series.yaml on the way (same workflow as 2210): the
patched labels + edu_patch_reason tracking column are written back to
the parquet so the override is visible on disk.
"""

import json
import math
from pathlib import Path

import pandas as pd

from patch_classifications import apply_patches

ROOT = Path(__file__).parent
CLASSIFIED = ROOT / "data" / "all_series_classified.parquet"
PATCHES = ROOT / "patches_all_series.yaml"
OUT = ROOT / "site" / "all_series_data.json"

SITE_COLUMNS = [
    "usajobs_control_number",
    "usajobs_url",
    "position_title",
    "agency",
    "department",
    "series_num",
    "series_title",
    "opm_tier",
    "opm_mandatory_type",
    "min_grade",
    "max_grade",
    "source_year",
    "edu_category",
    "edu_key_quote",
    "edu_patch_reason",
    "data_source",
    "is_public",
]


def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main():
    df = pd.read_parquet(CLASSIFIED).drop_duplicates("usajobs_control_number", keep="first")
    before_counts = df["edu_category"].value_counts().to_dict()
    patched = apply_patches(df, PATCHES)
    after_counts = patched["edu_category"].value_counts().to_dict()
    n_patched = int((patched.get("edu_patch_reason", pd.Series("", index=patched.index)) != "").sum())
    print(f"Applied {PATCHES.name}: {n_patched} rows flipped")
    if n_patched:
        print(f"  before: {before_counts}")
        print(f"  after:  {after_counts}")
    # Persist patched labels + tracking column back to disk (idempotent).
    patched.to_parquet(CLASSIFIED, index=False)
    print(f"Wrote patched parquet back to {CLASSIFIED.name}")

    site_df = patched[SITE_COLUMNS]
    records = site_df.to_dict(orient="records")
    for r in records:
        for k in r:
            r[k] = clean(r[k])
    OUT.write_text(json.dumps(records, default=str))
    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
