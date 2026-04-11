#!/usr/bin/env python3
"""
Validation: scrape USAJobs announcement pages for a sample of 2210 jobs that
are in BOTH historical_jobs and 2210_raw (so the API-supplied Education /
QualificationSummary fields are ground truth), extract the Education and
Qualifications sections from the HTML, and compare.

Purpose: decide whether regex-on-HTML is reliable enough to mirror the current
jobs pipeline, or whether we need to fall back to sending full announcement
text to the LLM.
"""

import io
import re
import time
import urllib.request
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "usajobs_html_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
R2 = "https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev/data"
UA = "Mozilla/5.0 (compatible; usajobs-education-research/0.1; +https://github.com/abigailhaddad)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def load_historical_2210() -> pd.DataFrame:
    frames = []
    for y in (2024, 2025, 2026):
        raw = fetch(f"{R2}/historical_jobs_{y}.parquet")
        df = pd.read_parquet(io.BytesIO(raw))
        df = df[df["JobCategories"].fillna("").str.contains("2210", na=False)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def scrape(cn: str) -> str:
    path = CACHE_DIR / f"{cn}.html"
    if path.exists():
        return path.read_text()
    html = fetch(f"https://www.usajobs.gov/job/{cn}").decode("utf-8", "replace")
    path.write_text(html)
    time.sleep(0.5)
    return html


# --- extraction ---------------------------------------------------------

# Strategy: find the h3 for a section, then grab sibling content up to the
# next h3/h2.
SECTION_RE = re.compile(
    r"<h3[^>]*>\s*(?P<name>[^<]+?)\s*</h3>(?P<body>.*?)(?=<h[23]\b)",
    re.S | re.I,
)


def html_to_text(html: str) -> str:
    # drop script/style
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    # turn block-level and br into newlines so paragraph structure survives
    html = re.sub(r"</(p|div|li|br|h[1-6]|tr)>", "\n", html, flags=re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = unescape(text)
    # collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_sections(html: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in SECTION_RE.finditer(html):
        name = m.group("name").strip().lower()
        if name in ("qualifications", "education"):
            out[name] = html_to_text(m.group("body"))
    return out


# --- similarity ---------------------------------------------------------


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def sim(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def coverage(truth: str, scraped: str) -> float:
    """What fraction of the ground-truth tokens also appear in scraped text.

    Sequence ratio penalizes reordering / extra navigation cruft; token
    coverage tells us whether the *content* is present.
    """
    t_toks = set(re.findall(r"[a-z0-9]{3,}", normalize(truth)))
    if not t_toks:
        return 1.0
    s_toks = set(re.findall(r"[a-z0-9]{3,}", normalize(scraped)))
    return len(t_toks & s_toks) / len(t_toks)


# --- driver -------------------------------------------------------------


def main():
    print("Loading historical 2210...")
    hist = load_historical_2210()
    hist["cn"] = hist["usajobsControlNumber"].astype(str)

    print("Loading current 2210_raw...")
    cur = pd.read_parquet(DATA_DIR / "2210_raw.parquet")
    cur["cn"] = cur["usajobs_control_number"].astype(str)

    overlap = sorted(set(cur["cn"]) & set(hist["cn"]))
    print(f"overlap: {len(overlap)} control numbers")

    sample = pd.Series(overlap).sample(20, random_state=7).tolist()
    cur_idx = cur.set_index("cn")

    rows = []
    for cn in sample:
        truth = cur_idx.loc[cn]
        if isinstance(truth, pd.DataFrame):
            truth = truth.iloc[0]
        try:
            html = scrape(cn)
        except Exception as e:
            print(f"  {cn}: fetch error {e}")
            continue
        sections = extract_sections(html)
        edu_scraped = sections.get("education", "")
        qual_scraped = sections.get("qualifications", "")
        edu_truth = str(truth.get("education") or "")
        qual_truth = str(truth.get("qualification_summary") or "")

        rows.append({
            "cn": cn,
            "edu_truth_len": len(edu_truth),
            "edu_scrape_len": len(edu_scraped),
            "edu_cov": round(coverage(edu_truth, edu_scraped), 3),
            "edu_sim": round(sim(edu_truth, edu_scraped), 3),
            "qual_truth_len": len(qual_truth),
            "qual_scrape_len": len(qual_scraped),
            "qual_cov": round(coverage(qual_truth, qual_scraped), 3),
            "qual_sim": round(sim(qual_truth, qual_scraped), 3),
        })

    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False))
    print()
    print("summary:")
    for col in ("edu_cov", "edu_sim", "qual_cov", "qual_sim"):
        print(f"  {col}: mean={res[col].mean():.3f}  min={res[col].min():.3f}")

    # Show worst cases
    print()
    print("worst education match:")
    worst = res.sort_values("edu_cov").iloc[0]
    cn = worst["cn"]
    print(f"  cn={cn}  cov={worst['edu_cov']}")
    truth = cur_idx.loc[cn]
    if isinstance(truth, pd.DataFrame):
        truth = truth.iloc[0]
    sc = extract_sections(scrape(cn))
    print("  --- API truth (first 400) ---")
    print(" ", normalize(str(truth.get("education") or ""))[:400])
    print("  --- scraped (first 400) ---")
    print(" ", normalize(sc.get("education", ""))[:400])


if __name__ == "__main__":
    main()
