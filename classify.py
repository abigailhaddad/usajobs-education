#!/usr/bin/env python3
"""
Classify education requirements for 2210 IT jobs using gpt-5.4-mini with structured output.

Usage:
    python classify.py                  # Classify all jobs
    python classify.py --sample 20      # Test on 20 random jobs first
    python classify.py --dry-run        # Show what would be sent without calling the API
    python classify.py --verify         # Verify quotes in already-classified data
"""

import argparse
import asyncio
import hashlib
import json
import time
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_raw.parquet"
HISTORICAL_INPUT = DATA_DIR / "2210_historical_raw.parquet"
OUTPUT = DATA_DIR / "2210_classified.parquet"
CACHE_FILE = DATA_DIR / "classification_cache.json"

MODEL = "openai/gpt-5.4-mini"
MAX_CONCURRENT = 20


class EducationCategory(str, Enum):
    no_education = "no_education"
    education_required = "education_required"
    education_required_higher = "education_required_higher"
    not_a_posting = "not_a_posting"


class EducationClassification(BaseModel):
    category: EducationCategory = Field(
        description="The education requirement category for this job posting"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this category was chosen"
    )
    key_quote: str = Field(
        description="The exact quote from the input text that most supports this classification. Must be copied verbatim — do not paraphrase. If the field is empty or the classification is based on absence of information, use '(no relevant text)'"
    )


SYSTEM_PROMPT = """You are classifying federal job postings (2210 IT Specialist series) by whether they have an education REQUIREMENT — meaning a degree you MUST have, with no experience-only alternative.

You will receive two fields from a USAJobs posting:
- "Education field": the dedicated education section of the posting
- "Qualifications Summary": the qualifications/requirements section

The actual education requirement may appear in EITHER field, or both, or neither. Read both carefully.

Classify into exactly one category:

- no_education: There is no grade level in this posting where a degree is strictly required. This is the most common case and includes ALL of the following:
  * Posting explicitly says no education requirement or no education substitution
  * Education can substitute for experience (but experience alone also works) — this is NOT an education requirement
  * Education is "preferred" or "desired" but not required
  * Education field is empty, boilerplate (transcript instructions, accreditation info), or contains overflow text from other sections (duties, admin info)
  * Only a high school diploma/GED is mentioned
  * OPM Alternative A or "individual occupational requirement" is referenced — this standard allows experience OR education, so there is no education requirement

- education_required: A degree (bachelor's or higher) is explicitly required at ALL grade levels in the posting, with no experience-only alternative. The posting must make clear you cannot qualify without the degree.

- education_required_higher: A degree is not required at lower grade levels, but IS required (with no experience alternative) at higher grade levels. For example: "GS-9 can qualify with experience, but GS-12 requires a bachelor's degree." Note: if the posting says education can substitute for experience at lower levels but is experience-only at higher levels, that is no_education — there is no grade where a degree is REQUIRED.

- not_a_posting: Not an actual job posting — a DHA public notice collecting resumes, a placeholder, or a notice that explicitly says it is not posted for applications.

KEY DISTINCTION: "education can substitute for experience" is NOT an education requirement. If you can qualify with experience alone at every grade level, that is no_education — even if the posting also offers a degree path. Common phrasings that are NOT education requirements:
- "If you are using Education to qualify for this position, [degree details]" — this is an optional education path
- "to qualify by education alone, [degree details]" — the word "alone" tells you experience is the other path
- "Ph.D. or equivalent doctoral degree or 3 full years of graduate education" at a specific grade level — this describes the education SUBSTITUTION option, not a requirement
- "In addition to the basic education requirements, you must have [degree]" in the context of the 2210 series — the "basic education requirement" for 2210 Alternative A can be met through experience OR education, so this is still no_education
- "you must have [degree] to qualify by education alone" — the phrase "by education alone" explicitly means there is an experience alternative. This is no_education.

EXAMPLE — this is no_education, NOT education_required:
"In addition to the basic education requirements, you must have a Ph.D. or equivalent doctoral degree or 3 full years of progressively higher level graduate education leading to a Ph.D. or equivalent doctoral degree to qualify by education alone."
This says "to qualify by education alone" — meaning experience is an alternative path. No degree is REQUIRED. This is no_education.

CRITICAL — "education is not substitutable" means NO education requirement:
- "Education is not substitutable for specialized experience at the GS-12 grade level" means you CANNOT use a degree to qualify — you MUST have experience. This is the OPPOSITE of an education requirement. This is no_education.
- "There is no substitution of education for experience at the GG-12 grade level" — same thing. No degree can help you. Experience only. This is no_education.
- These phrases mean the agency does NOT accept degrees as a way to qualify. They do NOT mean a degree is required.

RESOLVING CONTRADICTIONS:
- The Qualifications Summary often has generic boilerplate like "you may qualify with experience, education, or a combination." This is template language — do not treat it as establishing an education requirement.
- If the Education field explicitly says education CANNOT be substituted, that is authoritative. Do not override it with generic qual summary boilerplate.

IMPORTANT: Your key_quote must be copied EXACTLY from the input text — do not paraphrase or summarize."""


def build_prompt(education: str, qualification_summary: str) -> str:
    edu_text = education.strip() if education else "(empty)"
    qual_text = qualification_summary.strip() if qualification_summary else "(empty)"
    return f"""Education field:
{edu_text}

Qualifications Summary:
{qual_text}"""


def get_cache_key(education: str, qualification_summary: str) -> str:
    combined = (education or "") + "|" + (qualification_summary or "")
    return hashlib.sha256(combined.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


async def classify_one_async(
    education: str, qualification_summary: str, semaphore: asyncio.Semaphore
) -> EducationClassification:
    async with semaphore:
        user_msg = build_prompt(education, qualification_summary)
        resp = await acompletion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=EducationClassification,
        )
        return EducationClassification.model_validate_json(
            resp.choices[0].message.content
        )


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


async def run_classification(df: pd.DataFrame, cache: dict) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = [None] * len(df)
    cached_count = 0
    api_count = 0

    # Split into cached and uncached
    uncached_indices = []
    for i, (_, row) in enumerate(df.iterrows()):
        key = get_cache_key(row["education"], row["qualification_summary"])
        if key in cache:
            result = EducationClassification(**cache[key])
            results[i] = {
                "edu_category": result.category.value,
                "edu_reasoning": result.reasoning,
                "edu_key_quote": result.key_quote,
            }
            cached_count += 1
        else:
            uncached_indices.append(i)

    print(f"  {cached_count} cached, {len(uncached_indices)} need API calls")

    if not uncached_indices:
        return results

    # Process uncached in parallel with progress bar
    rows_list = list(df.iterrows())
    pbar = tqdm(total=len(uncached_indices), desc="Classifying", unit="job")
    save_interval = 100
    api_since_save = 0

    async def process_one(idx):
        nonlocal api_count, api_since_save
        _, row = rows_list[idx]
        try:
            result = await classify_one_async(
                row["education"], row["qualification_summary"], semaphore
            )
        except Exception as e:
            tqdm.write(f"  ERROR {row['usajobs_control_number']}: {e}")
            result = EducationClassification(
                category=EducationCategory.no_education,
                reasoning=f"API error: {e}",
                key_quote="(no relevant text)",
            )

        key = get_cache_key(row["education"], row["qualification_summary"])
        cache[key] = result.model_dump()
        results[idx] = {
            "edu_category": result.category.value,
            "edu_reasoning": result.reasoning,
            "edu_key_quote": result.key_quote,
        }
        api_count += 1
        api_since_save += 1
        pbar.update(1)

        # Save cache periodically
        if api_since_save >= save_interval:
            save_cache(cache)
            api_since_save = 0
            tqdm.write(f"  Cache saved ({api_count} API calls so far)")

    # Run in batches to avoid overwhelming
    batch_size = MAX_CONCURRENT * 2
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch = uncached_indices[batch_start : batch_start + batch_size]
        await asyncio.gather(*[process_one(idx) for idx in batch])

    pbar.close()
    save_cache(cache)
    print(f"  Final cache save ({cached_count} cached, {api_count} API calls)")

    return results


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
        # Historical scrape excludes CNs already in 2210_raw, but enforce
        # here too so re-running with partial state stays safe.
        api_cns = set(df["usajobs_control_number"].astype(str))
        hist = hist[~hist["usajobs_control_number"].astype(str).isin(api_cns)]
        df = pd.concat([df, hist], ignore_index=True)
        print(f"  +{len(hist)} historical scraped rows from {HISTORICAL_INPUT.name}")
        print(f"  total: {len(df)} jobs")

    if args.sample:
        df = df.sample(args.sample, random_state=42)
        print(f"Sampled {len(df)} jobs")

    if args.dry_run:
        for _, row in df.head(3).iterrows():
            print("=" * 60)
            print(build_prompt(row["education"], row["qualification_summary"]))
            print()
        return

    cache = load_cache()
    results = asyncio.run(run_classification(df, cache))

    result_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    out.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(out)} classified jobs to {OUTPUT}")
    print(f"\n{out['edu_category'].value_counts().to_string()}")

    # Auto-verify quotes
    bad = verify_quotes(out)
    if len(bad) == 0:
        print(f"\nAll quotes verified OK")
    else:
        print(f"\n{len(bad)} bad quotes (not found in source text):")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
