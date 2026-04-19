#!/usr/bin/env python3
"""
Pull whoMayApply / HiringPaths metadata for historical 2210 postings from the
R2 historical_jobs_YYYY.parquet files (which mirror the USAJobs HistoricJoa API).

Writes data/2210_historical_metadata.parquet keyed by usajobs_control_number
so prep_site_data.py can join audience info onto both API and scraped rows.
"""

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "2210_historical_metadata.parquet"
YEARS = [2023, 2024, 2025, 2026]


def load_parquet(year: int) -> pd.DataFrame:
    url = f"{R2_BASE}/data/historical_jobs_{year}.parquet"
    print(f"Fetching historical_jobs_{year}...")
    req = urllib.request.Request(url, headers={"User-Agent": "usajobs-download/1.0"})
    resp = urllib.request.urlopen(req)
    return pd.read_parquet(io.BytesIO(resp.read()))


def filter_2210(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["JobCategories"].astype(str).str.contains("2210", na=False)
    return df[mask].copy()


def hiring_paths_list(v) -> list[str]:
    """HiringPaths is stored as a JSON string of [{hiringPath: "..."}] or similar."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        raw = v
    elif isinstance(v, str):
        try:
            raw = json.loads(v)
        except Exception:
            return []
    else:
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            name = item.get("hiringPath") or item.get("HiringPath") or ""
            if name:
                out.append(name)
        elif isinstance(item, str):
            out.append(item)
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)
    frames = []
    for y in YEARS:
        df = load_parquet(y)
        sub = filter_2210(df)
        print(f"  {len(sub)} 2210 rows in {y}")
        frames.append(sub)

    allrows = pd.concat(frames, ignore_index=True)
    allrows["usajobs_control_number"] = allrows["usajobsControlNumber"].astype(str)

    # Normalize — is_public is true iff "The public" appears in HiringPaths.
    # Fall back to whoMayApply heuristic when HiringPaths is empty.
    hp_lists = allrows["HiringPaths"].apply(hiring_paths_list)
    is_public_by_path = hp_lists.apply(lambda lst: "The public" in lst if lst else None)
    wma_lower = allrows["whoMayApply"].fillna("").astype(str).str.lower()
    wma_public = wma_lower.str.contains(r"united states citizens|the public|all u\.s|open to the public", regex=True)
    wma_internal = wma_lower.str.contains(r"agency employees only|status|merit promotion|federal employees|internal to", regex=True)
    is_public_by_wma = wma_public.where(wma_public | wma_internal, None).where(~wma_internal, False)

    # Prefer HiringPaths signal, fall back to whoMayApply
    is_public = is_public_by_path.fillna(is_public_by_wma)

    out = pd.DataFrame({
        "usajobs_control_number": allrows["usajobs_control_number"],
        "who_may_apply": allrows["whoMayApply"],
        "hiring_paths": hp_lists,
        "is_public": is_public,
    }).drop_duplicates("usajobs_control_number", keep="first")

    out.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(out)} historical-metadata rows to {OUTPUT}")
    print(f"\nis_public distribution:")
    print(out["is_public"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
