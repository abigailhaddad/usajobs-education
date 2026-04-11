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
import random
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
MAX_CONCURRENT = 5
MAX_RETRIES = 5

# Bump whenever the verify prompt or category set changes — cache entries
# are keyed against this so stale verifications don't get returned.
PROMPT_VERSION = "v3-2026-04-11-mandatory-edge-case"

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
        description="The correct category. Must be one of: no_education, education_substitutable, education_required, education_required_higher, not_a_posting"
    )
    reasoning: str = Field(
        description="Brief explanation of why the original was correct or what was wrong"
    )
    key_quote: str = Field(
        description="The exact verbatim quote from the input that most supports the corrected classification. Copy it exactly — do not paraphrase."
    )


VERIFY_PROMPT = """You are reviewing a classification of a federal 2210 IT Specialist job posting's education requirements.

A first-pass model classified this posting. Your job is to verify whether the classification is correct.

The categories (mutually exclusive) are:

- no_education: Education is NOT a qualifying path at any grade. You MUST have experience — a degree alone will not get you in. Includes: postings that explicitly say "no substitution of education for experience", postings where the Education field is empty or only has transcript/boilerplate text, postings where only experience is described as a way to qualify, postings that only mention a high school diploma.

- education_substitutable: Education is NOT mandatory at any grade, but the posting explicitly offers education as ONE way to qualify — alongside experience. A candidate with the right degree and no experience CAN qualify because the posting allows education to substitute for experience (at some or all grades). This is the shape of most 2210 postings that reference OPM Alternative A: "you may qualify with IT-related experience OR a 4-year degree in computer science, information science, information systems management, mathematics, statistics, operations research, or engineering."

- education_required: A degree (bachelor's or higher) is required at ALL grade levels with no experience-only alternative.

- education_required_higher: A degree is required at higher grade levels (with no experience alternative at those grades), but NOT at the lowest grade level.

- not_a_posting: Not a real job posting (DHA notice, resume collection, placeholder).

CRITICAL DISTINCTIONS:

1. no_education vs education_substitutable — does the posting offer education as a qualifying PATH?
   - "If you are using Education to qualify..." / "to qualify by education alone..." → education_substitutable (education IS being offered as an alternative)
   - "Ph.D. or 3 full years of graduate education to qualify by education alone" → education_substitutable (the phrase explicitly means education alone is sufficient)
   - "Education is not substitutable for experience" at every grade → no_education (degree does not help)
   - Empty / boilerplate Education field with no education path in the qual summary → no_education
   - HS diploma only → no_education

2. education_substitutable vs education_required — is the degree MANDATORY?
   - "You may qualify with [experience] OR [degree]" → education_substitutable
   - "You MUST have a bachelor's degree..." with no experience-only path → education_required
   - "Positive education requirement" in 2210 context is ambiguous — it often refers to the OPM Alternative A standard that ALLOWS experience or education. Check whether the posting forecloses the experience path entirely before calling it education_required.

3. HARD EDGE CASE — "degree is mandatory" with a separate experience section.
   Some postings have Education field saying a degree is "mandatory" AND a Qualifications Summary listing required experience as a separate section. The question is whether these are ALTERNATIVES (one OR the other) or CUMULATIVE (both required).
   Test: look for explicit "OR" / "in lieu of" / "substitute" / "may qualify with" language linking them.
   - CUMULATIVE (no OR/in-lieu/substitute language) → education_required. Both the degree AND the experience are needed; the degree's "mandatory" label stands.
   - ALTERNATIVES (has OR/in-lieu/substitute language somewhere) → education_substitutable.
   Boilerplate like "If you are using education to qualify for this position, you must provide transcripts..." is NOT by itself evidence of substitutability — it's a standard transcript-submission directive. You need to see the degree presented as an ALTERNATIVE.

4. A degree is only "required" if the posting makes clear you CANNOT qualify without it at some grade level.

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
        PROMPT_VERSION
        + "|"
        + (str(row["education"]) if pd.notna(row["education"]) else "")
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


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    billing_markers = (
        "insufficient_quota",
        "insufficient quota",
        "exceeded your current quota",
        "check your plan and billing",
    )
    if any(m in s for m in billing_markers):
        return False
    return any(k in s for k in ("rate", "429", "timeout", "timed out", "connection", "overloaded"))


async def verify_one(row: pd.Series, semaphore: asyncio.Semaphore) -> VerificationResult:
    async with semaphore:
        system = VERIFY_PROMPT.format(
            original_category=row["edu_category"],
            original_reasoning=row.get("edu_reasoning", ""),
        )
        user_msg = build_verify_prompt(row)

        for attempt in range(MAX_RETRIES):
            try:
                resp = await acompletion(
                    model=VERIFY_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=VerificationResult,
                    temperature=0,
                )
                return VerificationResult.model_validate_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt == MAX_RETRIES - 1 or not _is_retryable(e):
                    raise
                delay = min(60, (2 ** attempt) + random.random())
                await asyncio.sleep(delay)
        raise RuntimeError("retry loop exited without returning or raising")


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
