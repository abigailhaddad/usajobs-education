#!/usr/bin/env python3
"""
Second-pass verification of education classifications using a stronger model.

Usage:
    python verify.py                # Run verification
    python verify.py --dry-run      # Show what would be sent
"""

import argparse
import asyncio
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_batch import CONFIG, call_llm, cache_key, load_cache, save_cache, run_batch

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_classified.parquet"
OUTPUT = DATA_DIR / "2210_verified.parquet"
CACHE_FILE = DATA_DIR / "verification_cache.json"

TASK = CONFIG["verify"]
MODEL = TASK["model"]
SYSTEM_PROMPT_TEMPLATE = TASK["system_prompt"]
PROMPT_VERSION = TASK.get("prompt_version", "")


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


# ── Schema ────────────────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    original_correct: bool = Field(
        description="True if the original classification is correct, False if it should be changed"
    )
    corrected_category: str = Field(
        description="The correct category. Must be one of: no_education, education_substitutable, "
                    "education_required, education_required_higher, not_a_posting"
    )
    reasoning: str = Field(
        description="Brief explanation of why the original was correct or what was wrong"
    )
    key_quote: str = Field(
        description="The exact verbatim quote from the input that most supports the corrected "
                    "classification. Copy it exactly — do not paraphrase."
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def needs_verification(row: pd.Series) -> str | None:
    """Returns reason for verification, or None if not needed."""
    cat = row["edu_category"]
    if cat != "no_education":
        return f"non-no_education: {cat}"

    combined = (str(row["education"]) if pd.notna(row["education"]) else "") + " " + (
        str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else ""
    )
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, combined):
            return f"suspicious pattern: {pat[:50]}"
    return None


def build_user_prompt(item: dict) -> str:
    edu = str(item.get("education") or "").strip() or "(empty)"
    qual = str(item.get("qualification_summary") or "").strip() or "(empty)"
    return f"Education field:\n{edu}\n\nQualifications Summary:\n{qual}"


def make_cache_key(item: dict) -> str:
    return cache_key(
        item.get("education") or "",
        item.get("qualification_summary") or "",
        item.get("edu_category") or "",
        version=PROMPT_VERSION,
    )


async def verify_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        system = SYSTEM_PROMPT_TEMPLATE.format(
            original_category=item.get("edu_category", ""),
            original_reasoning=item.get("edu_reasoning", ""),
        )
        result = await call_llm(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            response_format=VerificationResult,
        )
        return result.model_dump()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} classified jobs")

    verify_reasons = [needs_verification(row) for _, row in df.iterrows()]
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

    items = to_verify.to_dict(orient="records")
    cache = load_cache(CACHE_FILE)
    results, failed = asyncio.run(run_batch(
        items=items,
        process_fn=verify_one,
        cache=cache,
        cache_key_fn=make_cache_key,
        cache_path=CACHE_FILE,
        desc="Verifying",
        save_interval=50,
    ))

    if failed:
        print(f"\n{len(failed)} verifications failed — those rows keep their first-pass label.")

    # Apply corrections
    corrections = 0
    failed_set = set(failed)
    verify_indices = to_verify.index.tolist()
    for i, result in enumerate(results):
        if i in failed_set or result is None:
            continue
        idx = verify_indices[i]
        if not result["original_correct"]:
            df.loc[idx, "edu_category"] = result["corrected_category"]
            df.loc[idx, "edu_reasoning"] = result["reasoning"]
            df.loc[idx, "edu_key_quote"] = result["key_quote"]
            corrections += 1
        df.loc[idx, "verified"] = True
        df.loc[idx, "verification_agreed"] = result["original_correct"]

    df["verified"] = df["verified"].fillna(False)
    df["verification_agreed"] = df["verification_agreed"].fillna(True)
    df = df.drop(columns=["verify_reason"])
    df.to_parquet(OUTPUT, index=False)

    print(f"\n=== Verification complete ===")
    print(f"Verified: {len(to_verify)}")
    print(f"Corrections: {corrections}")
    print(f"\nFinal counts:")
    print(df["edu_category"].value_counts().to_string())

    if corrections > 0:
        changed = df[df["verified"] & ~df["verification_agreed"]]
        print(f"\n=== {len(changed)} corrections made ===")
        for _, r in changed.iterrows():
            print(f"  {r['usajobs_control_number']}: -> {r['edu_category']}")
            print(f"    {r['edu_reasoning'][:200]}")
            print()


if __name__ == "__main__":
    main()
