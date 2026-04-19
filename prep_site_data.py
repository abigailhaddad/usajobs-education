#!/usr/bin/env python3
"""
Build site/data.json from the classified + YOE parquet outputs.

Joins yoe labels onto the education-classified rows, keeps only the
fields the site renders, and writes JSON.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CLASSIFIED = ROOT / "data" / "2210_classified.parquet"
YOE = ROOT / "data" / "2210_yoe.parquet"
HIST_META = ROOT / "data" / "2210_historical_metadata.parquet"
OUT = ROOT / "site" / "data.json"

SITE_COLUMNS = [
    "usajobs_control_number",
    "usajobs_url",
    "position_title",
    "agency",
    "department",
    "min_grade",
    "max_grade",
    "source_year",
    "edu_category",
    "edu_key_quote",
    "data_source",
    "is_public",  # True=open to public, False=restricted, None=unknown (historical scraped)
    "yoe_categories",
    "yoe_quotes",
]


def main():
    edu = pd.read_parquet(CLASSIFIED).drop_duplicates("usajobs_control_number", keep="first")
    yoe = pd.read_parquet(YOE)[["usajobs_control_number", "yoe_categories", "yoe_quotes"]] \
        .drop_duplicates("usajobs_control_number", keep="first")
    meta = pd.read_parquet(HIST_META)[["usajobs_control_number", "is_public"]] \
        .drop_duplicates("usajobs_control_number", keep="first")
    print(f"Loaded {len(edu)} classified, {len(yoe)} YOE, {len(meta)} historical-metadata rows")

    df = edu.merge(yoe, on="usajobs_control_number", how="left")
    df = df.merge(meta, on="usajobs_control_number", how="left")

    # Rows with no YOE classification (edu_category = education_required or
    # not_a_posting) get an empty list — they're excluded from YOE analysis.
    df["yoe_categories"] = df["yoe_categories"].apply(
        lambda v: list(v) if v is not None and not (isinstance(v, float)) else []
    )

    df = df[SITE_COLUMNS]
    records = df.to_dict(orient="records")

    # Scrub NaN → None so json.dumps produces valid JSON (null, not NaN).
    # pandas to_dict can still leave float nan inside records.
    import math
    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    for r in records:
        for k in r:
            r[k] = clean(r[k])

    OUT.write_text(json.dumps(records, default=str))
    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
