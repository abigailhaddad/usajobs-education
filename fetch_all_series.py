#!/usr/bin/env python3
"""
Fetch a stratified random sample of GS postings across ALL occupational
series, 2023-2026. Output mirrors fetch_data.py's schema so classify.py /
verify.py work unchanged.

Pipeline:
  1. Sampling universe = historical_jobs_{2023..2026} (metadata for every
     posting that ever appeared on USAJobs in that window).
  2. Filter to GS postings whose series is in opm_series_tiers.json.
  3. Simple random sample on the pooled frame — that is proportional by
     (series × source_year) construction. Large series get many rows,
     small series get few. Fixed seed for reproducibility.
  4. Pull Education + QualificationSummary text:
       - If the CN is in current_jobs_{year}.parquet, use that text (free).
       - Otherwise scrape https://www.usajobs.gov/job/{CN} and regex out
         the Education + Qualifications <h3> sections (same code path as
         fetch_historical.py for 2210).
  5. Write data/all_series_raw.parquet.
"""

import argparse
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fetch_historical import extract_sections, fetch as http_fetch, UA

R2_BASE = "https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev/data"
REPO = Path(__file__).parent
DATA_DIR = REPO / "data"
TIERS_PATH = REPO / "opm_series_tiers.json"
OUTPUT = DATA_DIR / "all_series_raw.parquet"

HISTORICAL_YEARS = [2023, 2024, 2025, 2026]
CURRENT_YEARS = [2024, 2025, 2026]

RE_SERIES = re.compile(r'"series":\s*"(\d{4})"')


def load_parquet(url: str) -> pd.DataFrame:
    print(f"  {url.rsplit('/', 1)[-1]}")
    raw = http_fetch(url)
    return pd.read_parquet(io.BytesIO(raw))


def load_gs_series() -> dict[str, dict]:
    rows = json.loads(TIERS_PATH.read_text())
    out = {}
    for r in rows:
        num = (r.get("series_num") or "").strip().zfill(4)
        if not num:
            continue
        out[num] = {
            "series_title": r.get("series_title", ""),
            "opm_tier": r.get("tier", ""),
            "opm_mandatory_type": r.get("mandatory_type", ""),
        }
    return out


def parse_series(job_categories) -> list[str]:
    if job_categories is None or (isinstance(job_categories, float)):
        return []
    codes = RE_SERIES.findall(str(job_categories))
    return [c.zfill(4) for c in codes]


def parse_hiring_paths(raw) -> list[str]:
    """HiringPaths in both parquets is a JSON string of list[{'hiringPath': str}]."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        arr = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(arr, list):
        return []
    return [x.get("hiringPath", "") for x in arr if isinstance(x, dict)]


def build_universe(gs_series: dict) -> pd.DataFrame:
    """Load historical_jobs for all years and filter to GS + mapped series.

    Keep one row per (cn, source_year). A posting that spans year boundaries
    may appear in multiple year parquets; keep the earliest source_year.
    """
    frames = []
    print("Loading historical_jobs metadata:")
    for year in HISTORICAL_YEARS:
        df = load_parquet(f"{R2_BASE}/historical_jobs_{year}.parquet")
        df = df[df["payScale"] == "GS"].copy()
        df["source_year"] = year
        df["series_list"] = df["JobCategories"].apply(parse_series)
        # Primary series = first GS code on the posting that we have a tier for.
        def primary(codes):
            for c in codes:
                if c in gs_series:
                    return c
            return None
        df["series_num"] = df["series_list"].apply(primary)
        df = df[df["series_num"].notna()].copy()
        print(f"    {len(df):>7,} GS postings with mapped series in {year}")
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["cn"] = pool["usajobsControlNumber"].astype(str)
    pool = pool.sort_values("source_year").drop_duplicates("cn", keep="first").reset_index(drop=True)
    print(f"  → deduped universe: {len(pool):,} rows")
    return pool


def build_current_content(gs_series: dict) -> dict[str, dict]:
    """Load current_jobs and return {cn: {education, qualification_summary, hiring_path}}.

    Only used to avoid scraping — if a sampled CN is here, use this content
    directly. Historical metadata still provides the other fields.
    """
    content: dict[str, dict] = {}
    print("Loading current_jobs content:")
    for year in CURRENT_YEARS:
        df = load_parquet(f"{R2_BASE}/current_jobs_{year}.parquet")
        print(f"    {len(df):>7,} rows in current_jobs_{year}")
        for _, row in df.iterrows():
            try:
                d = json.loads(row["MatchedObjectDescriptor"])
            except (json.JSONDecodeError, TypeError):
                continue
            cn = str(row.get("usajobsControlNumber") or row.get("usajobs_control_number") or "")
            if not cn or cn in content:
                continue
            details = d.get("UserArea", {}).get("Details", {})
            content[cn] = {
                "education": details.get("Education", "") or "",
                "qualification_summary": d.get("QualificationSummary", "") or "",
            }
    print(f"  → {len(content):,} CNs with API content")
    return content


def meta_to_row(meta: pd.Series, gs_series: dict, education: str,
                qualification_summary: str, data_source: str) -> dict:
    cn = str(meta["usajobsControlNumber"])

    def s(key: str) -> str:
        v = meta.get(key)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    primary = meta["series_num"]
    tier_meta = gs_series.get(primary, {})
    hp = parse_hiring_paths(meta.get("HiringPaths"))

    return {
        "usajobs_control_number": cn,
        "usajobs_url": f"https://www.usajobs.gov/job/{cn}",
        "position_title": s("positionTitle"),
        "agency": s("hiringAgencyName"),
        "department": s("hiringDepartmentName"),
        "series_num": primary,
        "series_title": tier_meta.get("series_title", ""),
        "opm_tier": tier_meta.get("opm_tier", ""),
        "opm_mandatory_type": tier_meta.get("opm_mandatory_type", ""),
        "all_series_on_posting": meta["series_list"],
        "min_grade": s("minimumGrade"),
        "max_grade": s("maximumGrade"),
        "pay_scale": s("payScale"),
        "min_salary": s("minimumSalary"),
        "max_salary": s("maximumSalary"),
        "open_date": s("positionOpenDate"),
        "close_date": s("positionCloseDate"),
        "service_type": s("serviceType"),
        "appointment_type": s("appointmentType"),
        "hiring_path": hp,
        "is_public": "public" in [p.lower() for p in hp] or "The public" in hp,
        "education": education,
        "qualification_summary": qualification_summary,
        "source_year": int(meta["source_year"]),
        "data_source": data_source,
    }


def scrape_sample(sample: pd.DataFrame, current_content: dict[str, dict],
                  gs_series: dict, delay: float, timeout: int) -> list[dict]:
    rows: list[dict] = []
    errors = 0
    error_samples: list[str] = []
    n_api = 0
    n_scraped = 0

    pbar = tqdm(sample.iterrows(), total=len(sample), unit="job")
    for i, (_, meta) in enumerate(pbar):
        cn = str(meta["usajobsControlNumber"])
        if cn in current_content:
            c = current_content[cn]
            rows.append(meta_to_row(meta, gs_series, c["education"],
                                    c["qualification_summary"], "api"))
            n_api += 1
            continue

        try:
            html = http_fetch(f"https://www.usajobs.gov/job/{cn}", timeout=timeout).decode(
                "utf-8", "replace"
            )
            sec = extract_sections(html)
            rows.append(meta_to_row(meta, gs_series, sec["education"],
                                    sec["qualifications"], "scraped"))
            n_scraped += 1
        except urllib.error.HTTPError as e:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append(f"{cn}: HTTP {e.code}")
            if e.code in (404, 410):
                rows.append(meta_to_row(meta, gs_series, "", "", "scraped"))
        except Exception as e:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append(f"{cn}: {type(e).__name__}: {e}")

        if delay:
            time.sleep(delay)

    print(f"\n  {n_api} from current_jobs API, {n_scraped} scraped, {errors} errors")
    if error_samples:
        print("  First errors:")
        for s in error_samples:
            print(f"    {s}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", type=int, default=100,
                        help="Total sample size across all (series × year) strata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between scrape requests")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    gs_series = load_gs_series()
    print(f"Loaded {len(gs_series)} GS series from {TIERS_PATH.name}\n")

    universe = build_universe(gs_series)
    current_content = build_current_content(gs_series)

    n = min(args.sample_size, len(universe))
    print(f"\nSampling {n} rows (random_state={args.seed}) — "
          f"proportional to (series × year) by construction")
    sample = universe.sample(n=n, random_state=args.seed).reset_index(drop=True)

    by_year = sample["source_year"].value_counts().sort_index()
    n_series = sample["series_num"].nunique()
    overlap = sum(1 for cn in sample["cn"] if cn in current_content)
    print(f"  by source_year: {by_year.to_dict()}")
    print(f"  distinct series sampled: {n_series}")
    print(f"  covered by current_jobs API (no scrape needed): {overlap}")
    print(f"  will scrape: {len(sample) - overlap}\n")

    rows = scrape_sample(sample, current_content, gs_series,
                         delay=args.delay, timeout=args.timeout)
    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT}")
    print(f"  source breakdown: {out['data_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
