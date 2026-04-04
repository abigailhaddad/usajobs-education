#!/usr/bin/env python3
"""
Second-pass verification of education classifications using a stronger model.

Sends jobs to gpt-5.4 (full, not mini) for review when:
1. The first pass classified them as anything other than no_education
2. The first pass classified them as no_education but the text contains
   patterns that might indicate a missed education requirement

Usage:
    python verify.py                # Run verification
    python verify.py --dry-run      # Show what would be sent
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_classified.parquet"
OUTPUT = DATA_DIR / "2210_verified.parquet"
VERIFY_CACHE = DATA_DIR / "verification_cache.json"

VERIFY_MODEL = "openai/gpt-5.4"
MAX_CONCURRENT = 10

# Regex patterns that flag a no_education classification for review
SUSPICIOUS_PATTERNS = [
    r'(?i)\bmandatory\b.{0,50}(degree|bachelor|education)',
    r'(?i)must\s+(possess|have)\s+a\s+(bachelor|master|degree)',
    r'(?i)(degree|bachelor).{0,30}(required|mandatory)',
    r'(?i)positive\s+education\s+requirement',
    r'(?i)minimum\s+education(al)?\s+requirement.{0,50}(bachelor|degree)',
    r'(?i)degree\s+is\s+(required|mandatory)',
    r'(?i)(require[sd])\s+a\s+(bachelor|master|college|undergraduate)',
    r'(?i)an?\s+undergraduate\s+degree.{0,30}is\s+mandatory',
]


class VerificationResult(BaseModel):
    original_correct: bool = Field(
        description="True if the original classification is correct, False if it should be changed"
    )
    corrected_category: str = Field(
        description="The correct category. Must be one of: no_education, education_required, education_required_higher, not_a_posting"
    )
    reasoning: str = Field(
        description="Brief explanation of why the original was correct or what was wrong"
    )
    key_quote: str = Field(
        description="The exact verbatim quote from the input that most supports the corrected classification. Copy it exactly — do not paraphrase."
    )


VERIFY_PROMPT = """You are reviewing a classification of a federal 2210 IT Specialist job posting's education requirements.

A first-pass model classified this posting. Your job is to verify whether the classification is correct.

The categories are:
- no_education: No grade level requires a degree. This includes: education can substitute for experience (not a requirement), education preferred/desired, empty/boilerplate education fields, high school diploma only, OPM Alternative A (experience OR education).
- education_required: A degree (bachelor's or higher) is required at ALL grade levels with no experience-only alternative.
- education_required_higher: A degree is required at higher grade levels but not lower ones — AND there is no experience-only path at those higher grades.
- not_a_posting: Not a real job posting (DHA notice, resume collection, placeholder).

CRITICAL DISTINCTIONS:
- "Education is not substitutable for experience" means you CANNOT use a degree — experience is required. This is no_education, NOT education_required.
- "If you are using Education to qualify..." or "to qualify by education alone..." means education is OPTIONAL. This is no_education.
- "Ph.D. or 3 full years of graduate education" at a grade level is an education SUBSTITUTION option, not a requirement. This is no_education.
- "Positive education requirement" in the 2210 series usually refers to OPM Alternative A, which allows experience OR education. This is typically no_education.
- A degree is only "required" if the posting makes clear you CANNOT qualify without it at some grade level.

The first-pass classification was: {original_category}
The first-pass reasoning was: {original_reasoning}

Review the actual posting text below and determine if the classification is correct."""


def needs_verification(row: pd.Series) -> str | None:
    """Returns reason for verification, or None if not needed."""
    cat = row["edu_category"]

    # All non-no_education get verified
    if cat != "no_education":
        return f"non-no_education: {cat}"

    # Check suspicious patterns in no_education
    combined = (str(row["education"]) if pd.notna(row["education"]) else "") + " " + (
        str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else ""
    )
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, combined):
            return f"suspicious pattern: {pat[:50]}"

    return None


def build_verify_prompt(row: pd.Series) -> str:
    edu = str(row["education"]).strip() if pd.notna(row["education"]) else "(empty)"
    qual = str(row["qualification_summary"]).strip() if pd.notna(row["qualification_summary"]) else "(empty)"
    return f"""Education field:
{edu}

Qualifications Summary:
{qual}"""


def get_verify_cache_key(row: pd.Series) -> str:
    combined = (
        (str(row["education"]) if pd.notna(row["education"]) else "")
        + "|"
        + (str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else "")
        + "|"
        + row["edu_category"]
    )
    return hashlib.sha256(combined.encode()).hexdigest()


def load_verify_cache() -> dict:
    if VERIFY_CACHE.exists():
        return json.loads(VERIFY_CACHE.read_text())
    return {}


def save_verify_cache(cache: dict):
    VERIFY_CACHE.write_text(json.dumps(cache, indent=2))


async def verify_one(row: pd.Series, semaphore: asyncio.Semaphore) -> VerificationResult:
    async with semaphore:
        system = VERIFY_PROMPT.format(
            original_category=row["edu_category"],
            original_reasoning=row.get("edu_reasoning", ""),
        )
        user_msg = build_verify_prompt(row)

        resp = await acompletion(
            model=VERIFY_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            response_format=VerificationResult,
        )
        return VerificationResult.model_validate_json(resp.choices[0].message.content)


async def run_verification(to_verify: pd.DataFrame, cache: dict) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = [None] * len(to_verify)
    cached_count = 0
    uncached_indices = []

    rows_list = list(to_verify.iterrows())
    for i, (_, row) in enumerate(rows_list):
        key = get_verify_cache_key(row)
        if key in cache:
            results[i] = cache[key]
            cached_count += 1
        else:
            uncached_indices.append(i)

    print(f"  {cached_count} cached, {len(uncached_indices)} need API calls")

    if not uncached_indices:
        return results

    pbar = tqdm(total=len(uncached_indices), desc="Verifying", unit="job")
    api_since_save = 0

    async def process_one(idx):
        nonlocal api_since_save
        _, row = rows_list[idx]
        try:
            result = await verify_one(row, semaphore)
            result_dict = result.model_dump()
        except Exception as e:
            tqdm.write(f"  ERROR {row['usajobs_control_number']}: {e}")
            result_dict = {
                "original_correct": True,
                "corrected_category": row["edu_category"],
                "reasoning": f"Verification error: {e}",
                "key_quote": "(no relevant text)",
            }

        key = get_verify_cache_key(row)
        cache[key] = result_dict
        results[idx] = result_dict
        api_since_save += 1
        pbar.update(1)

        if api_since_save >= 50:
            save_verify_cache(cache)
            api_since_save = 0

    batch_size = MAX_CONCURRENT * 2
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch = uncached_indices[batch_start : batch_start + batch_size]
        await asyncio.gather(*[process_one(idx) for idx in batch])

    pbar.close()
    save_verify_cache(cache)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} classified jobs")

    # Find jobs that need verification
    verify_reasons = []
    for _, row in df.iterrows():
        reason = needs_verification(row)
        verify_reasons.append(reason)
    df["verify_reason"] = verify_reasons

    to_verify = df[df["verify_reason"].notna()].copy()
    print(f"\n{len(to_verify)} jobs flagged for verification:")
    print(f"  Non-no_education: {(to_verify['edu_category'] != 'no_education').sum()}")
    print(f"  Suspicious no_education: {(to_verify['edu_category'] == 'no_education').sum()}")

    if args.dry_run:
        print("\nWould verify:")
        for cat in to_verify["edu_category"].unique():
            n = (to_verify["edu_category"] == cat).sum()
            print(f"  {cat}: {n}")
        return

    cache = load_verify_cache()
    results = asyncio.run(run_verification(to_verify, cache))

    # Apply corrections
    corrections = 0
    verify_indices = to_verify.index.tolist()
    for i, result in enumerate(results):
        idx = verify_indices[i]
        if not result["original_correct"]:
            old = df.loc[idx, "edu_category"]
            new = result["corrected_category"]
            df.loc[idx, "edu_category"] = new
            df.loc[idx, "edu_reasoning"] = result["reasoning"]
            df.loc[idx, "edu_key_quote"] = result["key_quote"]
            corrections += 1
        df.loc[idx, "verified"] = True
        df.loc[idx, "verification_agreed"] = result["original_correct"]

    # Mark unverified jobs
    df["verified"] = df["verified"].fillna(False)
    df["verification_agreed"] = df["verification_agreed"].fillna(True)

    df = df.drop(columns=["verify_reason"])
    df.to_parquet(OUTPUT, index=False)

    print(f"\n=== Verification complete ===")
    print(f"Verified: {len(to_verify)}")
    print(f"Corrections: {corrections}")
    print(f"\nFinal counts:")
    print(df["edu_category"].value_counts().to_string())

    # Show corrections
    if corrections > 0:
        changed = df[df["verified"] & ~df["verification_agreed"]]
        print(f"\n=== {len(changed)} corrections made ===")
        for _, r in changed.iterrows():
            print(f"  {r['usajobs_control_number']}: -> {r['edu_category']}")
            print(f"    {r['edu_reasoning'][:200]}")
            print()


if __name__ == "__main__":
    main()
