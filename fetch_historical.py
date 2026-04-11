#!/usr/bin/env python3
"""
Scrape USAJobs announcement pages for 2210 jobs that appear in the
historical_jobs parquets but are NOT in our current 2210_raw.parquet
(mostly older 2024 postings that rolled out of the current_jobs API
before we started pulling).

The historical parquets only carry metadata, so we fetch each
https://www.usajobs.gov/job/{control_number} page, extract the
Education and Qualifications sections from the HTML, and write a
parquet in the EXACT schema produced by fetch_data.py so classify.py
and verify.py work unchanged.

Usage:
    python fetch_historical.py                # full run (resumes from existing output)
    python fetch_historical.py --limit 50     # dry run — scrape only 50 jobs
    python fetch_historical.py --delay 1.0    # politer rate limit
"""

import argparse
import io
import re
import time
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
R2_BASE = "https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev/data"
CURRENT_RAW = DATA_DIR / "2210_raw.parquet"
OUTPUT = DATA_DIR / "2210_historical_raw.parquet"

UA = (
    "Mozilla/5.0 (compatible; usajobs-education-research/0.1; "
    "+https://github.com/abigailhaddad/usajobs-education)"
)

# Match an <h3>NAME</h3> and capture everything up to the next h2 or h3.
# The announcement page lays Education and Qualifications out as sibling
# h3s under the "Requirements" h2, so this is reliable.
SECTION_RE = re.compile(
    r"<h3[^>]*>\s*(?P<name>[^<]+?)\s*</h3>(?P<body>.*?)(?=<h[23]\b)",
    re.S | re.I,
)


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_historical_2210() -> pd.DataFrame:
    frames = []
    for year in (2024, 2025, 2026):
        print(f"  downloading historical_jobs_{year}.parquet")
        raw = fetch(f"{R2_BASE}/historical_jobs_{year}.parquet")
        df = pd.read_parquet(io.BytesIO(raw))
        df = df[df["JobCategories"].fillna("").str.contains("2210", na=False)].copy()
        df["source_year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def html_to_text(html: str) -> str:
    """Flatten announcement-section HTML to the same plain-text shape the
    USAJobs API returns in UserArea.Details.Education / QualificationSummary."""
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"</(p|div|li|br|h[1-6]|tr)>", "\n", html, flags=re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_sections(html: str) -> dict[str, str]:
    out = {"education": "", "qualifications": ""}
    for m in SECTION_RE.finditer(html):
        name = m.group("name").strip().lower()
        if name in out:
            out[name] = html_to_text(m.group("body"))
    return out


def scrape_one(cn: str, timeout: int = 30) -> dict[str, str]:
    html = fetch(f"https://www.usajobs.gov/job/{cn}", timeout=timeout).decode(
        "utf-8", "replace"
    )
    return extract_sections(html)


def row_from_meta(meta: pd.Series, education: str, qualification_summary: str) -> dict:
    cn = str(meta["usajobsControlNumber"])

    def s(key: str) -> str:
        v = meta.get(key)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    return {
        "usajobs_control_number": cn,
        "usajobs_url": f"https://www.usajobs.gov/job/{cn}",
        "position_title": s("positionTitle"),
        "agency": s("hiringAgencyName"),
        "department": s("hiringDepartmentName"),
        "min_grade": s("minimumGrade"),
        "max_grade": s("maximumGrade"),
        "pay_scale": s("payScale"),
        "min_salary": s("minimumSalary"),
        "max_salary": s("maximumSalary"),
        "open_date": s("positionOpenDate"),
        "close_date": s("positionCloseDate"),
        "service_type": s("serviceType"),
        "appointment_type": s("appointmentType"),
        "education": education,
        "qualification_summary": qualification_summary,
        "source_year": int(meta["source_year"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scrape N jobs (dry run)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests (default 0.5)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Write parquet every N successful scrapes")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print("Loading historical 2210 metadata from R2...")
    hist = load_historical_2210()
    hist["cn"] = hist["usajobsControlNumber"].astype(str)
    print(f"  {len(hist)} historical 2210 rows")

    skip_cns: set[str] = set()
    if CURRENT_RAW.exists():
        cur = pd.read_parquet(CURRENT_RAW)
        skip_cns |= set(cur["usajobs_control_number"].astype(str))
        print(f"  excluding {len(skip_cns)} already in {CURRENT_RAW.name}")
    else:
        print(f"  {CURRENT_RAW.name} not present — not excluding anything. "
              "Run fetch_data.py first if you want to avoid re-scraping current jobs.")

    existing_rows: list[dict] = []
    if OUTPUT.exists():
        ex = pd.read_parquet(OUTPUT)
        existing_cns = set(ex["usajobs_control_number"].astype(str))
        skip_cns |= existing_cns
        existing_rows = ex.to_dict(orient="records")
        print(f"  resuming — {len(existing_cns)} already scraped in {OUTPUT.name}")

    todo = hist[~hist["cn"].isin(skip_cns)].drop_duplicates("cn").reset_index(drop=True)
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"  {len(todo)} remaining to scrape")
    if len(todo) == 0:
        return

    rows = list(existing_rows)
    errors = 0
    error_samples: list[str] = []
    pbar = tqdm(todo.iterrows(), total=len(todo), unit="job")

    for i, (_, meta) in enumerate(pbar):
        cn = str(meta["usajobsControlNumber"])
        try:
            sections = scrape_one(cn, timeout=args.timeout)
            rows.append(
                row_from_meta(meta, sections["education"], sections["qualifications"])
            )
        except urllib.error.HTTPError as e:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append(f"{cn}: HTTP {e.code}")
            # 404/410: announcement has been fully removed — record empty so we
            # do not keep retrying on future runs.
            if e.code in (404, 410):
                rows.append(row_from_meta(meta, "", ""))
        except Exception as e:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append(f"{cn}: {type(e).__name__}: {e}")
            # transient — do NOT write a row, so next run retries this cn.

        if (i + 1) % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
            tqdm.write(f"  checkpoint: {len(rows)} rows written, {errors} errors")

        if args.delay:
            time.sleep(args.delay)

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(rows)} rows to {OUTPUT} ({errors} errors)")
    if error_samples:
        print("First errors:")
        for s in error_samples:
            print(f"  {s}")


if __name__ == "__main__":
    main()
