#!/usr/bin/env python3
"""
Classify education requirements for 2210 IT jobs.

Usage:
    python classify.py                  # Classify all jobs
    python classify.py --sample 20      # Test on 20 random jobs first
    python classify.py --dry-run        # Show what would be sent without calling the API
    python classify.py --verify         # Verify quotes in already-classified data
"""

import argparse
import asyncio
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_batch import CONFIG, call_llm, cache_key, load_cache, save_cache, run_batch

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_raw.parquet"
HISTORICAL_INPUT = DATA_DIR / "2210_historical_raw.parquet"
OUTPUT = DATA_DIR / "2210_classified.parquet"
CACHE_FILE = DATA_DIR / "classification_cache.json"

TASK = CONFIG["classify"]
MODEL = TASK["model"]
SYSTEM_PROMPT = TASK["system_prompt"]
PROMPT_VERSION = TASK.get("prompt_version", "")


# ── Schema ────────────────────────────────────────────────────────────────

class EducationCategory(str, Enum):
    no_education = "no_education"
    education_substitutable = "education_substitutable"
    education_required = "education_required"
    not_a_posting = "not_a_posting"


class EducationClassification(BaseModel):
    category: EducationCategory = Field(
        description="The education requirement category for this job posting"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this category was chosen"
    )
    key_quote: str = Field(
        description="The exact quote from the input text that most supports this classification. "
                    "Must be copied verbatim — do not paraphrase. If the field is empty or the "
                    "classification is based on absence of information, use '(no relevant text)'"
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def build_prompt(education: str, qualification_summary: str) -> str:
    edu_text = education.strip() if education else "(empty)"
    qual_text = qualification_summary.strip() if qualification_summary else "(empty)"
    return f"Education field:\n{edu_text}\n\nQualifications Summary:\n{qual_text}"


def make_cache_key(item: dict) -> str:
    return cache_key(
        item.get("education") or "",
        item.get("qualification_summary") or "",
        version=PROMPT_VERSION,
    )


async def classify_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        result = await call_llm(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(
                    item.get("education", ""),
                    item.get("qualification_summary", ""),
                )},
            ],
            response_format=EducationClassification,
        )
        return {
            "edu_category": result.category.value,
            "edu_reasoning": result.reasoning,
            "edu_key_quote": result.key_quote,
        }


def verify_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Check that key_quote actually appears in the source text."""
    bad = []
    for _, row in df.iterrows():
        quote = row.get("edu_key_quote", "")
        if not quote or quote == "(no relevant text)":
            continue
        combined = (str(row["education"]) if pd.notna(row["education"]) else "") + " " + (
            str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else ""
        )
        if quote not in combined:
            bad.append({
                "control_number": row["usajobs_control_number"],
                "category": row["edu_category"],
                "quote": quote[:150],
                "found": False,
            })
    return pd.DataFrame(bad) if bad else pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Classify a random sample of N jobs")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    parser.add_argument("--verify", action="store_true", help="Verify quotes in existing output")
    args = parser.parse_args()

    if args.verify:
        if not OUTPUT.exists():
            print(f"No output file at {OUTPUT}")
            return
        df = pd.read_parquet(OUTPUT)
        bad = verify_quotes(df)
        if len(bad) == 0:
            print(f"All {len(df)} quotes verified OK")
        else:
            print(f"{len(bad)} bad quotes out of {len(df)}:")
            print(bad.to_string(index=False))
        return

    df = pd.read_parquet(INPUT)
    df["data_source"] = "api"
    print(f"Loaded {len(df)} jobs from {INPUT.name}")
    if HISTORICAL_INPUT.exists():
        hist = pd.read_parquet(HISTORICAL_INPUT)
        hist["data_source"] = "scraped"
        api_cns = set(df["usajobs_control_number"].astype(str))
        hist = hist[~hist["usajobs_control_number"].astype(str).isin(api_cns)]
        df = pd.concat([df, hist], ignore_index=True)
        print(f"  +{len(hist)} historical scraped rows from {HISTORICAL_INPUT.name}")
        print(f"  total: {len(df)} jobs")

    # Coerce mixed-type columns (historical scraped = str, current API = float)
    for col in ("min_salary", "max_salary", "min_grade", "max_grade"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    if args.sample:
        df = df.sample(args.sample, random_state=42)
        print(f"Sampled {len(df)} jobs")

    if args.dry_run:
        for _, row in df.head(3).iterrows():
            print("=" * 60)
            print(build_prompt(row["education"], row["qualification_summary"]))
            print()
        return

    items = df.to_dict(orient="records")
    cache = load_cache(CACHE_FILE)
    results, failed = asyncio.run(run_batch(
        items=items,
        process_fn=classify_one,
        cache=cache,
        cache_key_fn=make_cache_key,
        cache_path=CACHE_FILE,
        desc="Classifying",
    ))

    if failed:
        print(f"\n{len(failed)} rows failed — dropping from output. Re-run to retry.")
        keep = [i not in set(failed) for i in range(len(df))]
        df = df.iloc[keep].reset_index(drop=True)
        results = [r for i, r in enumerate(results) if keep[i]]

    result_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    out.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(out)} classified jobs to {OUTPUT}")
    print(f"\n{out['edu_category'].value_counts().to_string()}")

    bad = verify_quotes(out)
    if len(bad) == 0:
        print(f"\nAll quotes verified OK")
    else:
        print(f"\n{len(bad)} bad quotes (not found in source text):")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
