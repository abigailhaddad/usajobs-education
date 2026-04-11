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
import random
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
MAX_CONCURRENT = 5
MAX_RETRIES = 5


class EducationCategory(str, Enum):
    no_education = "no_education"
    education_substitutable = "education_substitutable"
    education_required = "education_required"
    education_required_higher = "education_required_higher"
    not_a_posting = "not_a_posting"


# Bump this string whenever the prompt or category set changes so the
# text-keyed classification cache invalidates cleanly — old entries stay
# in the file but are no longer addressable.
PROMPT_VERSION = "v3-2026-04-11-mandatory-edge-case"


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


SYSTEM_PROMPT = """You are classifying federal job postings (2210 IT Specialist series) by the role education plays in qualifying for the position.

You will receive two fields from a USAJobs posting:
- "Education field": the dedicated education section of the posting
- "Qualifications Summary": the qualifications/requirements section

The actual requirements may appear in EITHER field, or both, or neither. Read both carefully.

Classify into exactly one category (mutually exclusive):

- education_required: A degree (bachelor's or higher) is explicitly required at ALL grade levels in the posting, with no experience-only alternative. The posting must make clear you cannot qualify without the degree.

- education_required_higher: A degree is required at one or more HIGHER grade levels (with no experience alternative at those grades), but NOT required at the lowest grade level. For example: "GS-9 can qualify with 1 year of specialized experience, but GS-12 requires a bachelor's degree with no experience alternative."

- education_substitutable: Education is NOT mandatory at any grade, but the posting explicitly offers education as ONE way to qualify — alongside experience. A candidate with the right degree and no relevant experience CAN qualify because the posting allows education to substitute for experience (at some or all grades). This is the shape of most 2210 postings that reference OPM Alternative A / the 2210 individual occupational requirement: "you may qualify with IT-related experience OR a 4-year degree in computer science, information science, information systems management, mathematics, statistics, operations research, or engineering."

- no_education: Education is NOT a qualifying path at any grade. To qualify at every grade level in this posting, you MUST have experience — a degree alone will not get you in. This includes:
  * Postings that explicitly say "no substitution of education for experience" or "education is not substitutable"
  * Postings where only experience is listed as a way to qualify and education is not mentioned as an alternative
  * Postings where the Education field is empty, boilerplate (transcript instructions, accreditation info, overflow text from other sections), or mentions only high school / GED
  * Postings where education is mentioned only as "preferred" or "desired" — that is not offering education as a QUALIFYING path

- not_a_posting: Not an actual job posting — a DHA public notice collecting resumes, a placeholder, or a notice that explicitly says it is not posted for applications.

KEY DISTINCTIONS

1. education_substitutable vs no_education — does the posting OFFER education as a qualifying path?
   - "Ph.D. or equivalent doctoral degree or 3 full years of graduate education to qualify by education alone" at a given grade → education_substitutable (the phrase "by education alone" explicitly means experience is the other path AND education alone is sufficient)
   - "Undergraduate and Graduate Education. Major study--computer science, information science, information systems management..." listed in the Education field → education_substitutable (the posting is telling you what degrees count for the education path)
   - "Specialized experience: 1 year at the next lower grade... OR Education: bachelor's degree with a major in..." → education_substitutable
   - "In lieu of specialized experience, you may have..." / "SUBSTITUTION OF EDUCATION FOR SPECIALIZED EXPERIENCE" → education_substitutable
   - "Education is not substitutable for specialized experience at the GS-12 grade level" at EVERY grade → no_education
   - "There is no substitution of education for experience" → no_education
   - Education field is empty or only contains boilerplate like transcript-submission instructions → no_education
   - Education field only mentions high school / GED → no_education
   - Boilerplate like "If you are using education to qualify for this position, College transcripts must be submitted" is NOT by itself evidence of substitutability — it's a standard transcript-submission directive that appears on postings regardless of whether education is actually a qualifying path. You need to see the degree presented as an ALTERNATIVE (OR / in lieu / substitute / "may qualify with") elsewhere in the text.

2. education_substitutable vs education_required — is the degree MANDATORY?
   - "You may qualify with [experience] OR [degree]" → education_substitutable (either path works)
   - "You MUST have a bachelor's degree in computer science to qualify" → education_required (no experience path)
   - "This position has a positive education requirement — a 4-year degree is required" at all grades → education_required
   - "Positive education requirement" is NOT automatically education_required. In 2210 context it often refers to the OPM standard that allows experience OR education. Look at whether the posting actually forecloses the experience path.

3. HARD EDGE CASE — "degree is mandatory" with a separate experience section.

   Some postings list BOTH a mandatory degree in the Education field AND required specialized experience in the Qualifications Summary as SEPARATE sections. The question is whether these are ALTERNATIVES (you need one OR the other) or CUMULATIVE (you need BOTH).

   The test: look for explicit "OR" / "in lieu of" / "substitute" / "may qualify with" language linking the two. If that language is absent, the degree AND experience are both required → education_required.

   CUMULATIVE (both required) → education_required:
     Education field: "An undergraduate degree in computer science is mandatory. The degree must be in..."
     Qual summary:    "GENERAL EXPERIENCE: ... SPECIALIZED EXPERIENCE: You must have 1 year of experience at the next lower grade."
     (No OR / in lieu / substitute language anywhere. You need BOTH the degree and the experience. Degree is mandatory → education_required.)

   ALTERNATIVES (education substitutes for experience) → education_substitutable:
     "You may qualify with 1 year of specialized experience OR a bachelor's degree in..."
     "In lieu of specialized experience, you may have..."
     "SUBSTITUTION OF EDUCATION FOR SPECIALIZED EXPERIENCE: ..."

   When in doubt on this edge case: if the Education field says "mandatory" / "required" and you can't find OR/in-lieu/substitute language connecting it to the experience requirements, classify as education_required.

4. education_required_higher vs education_substitutable at higher grades
   - If lower grades allow experience-or-education AND higher grades ALSO allow experience-or-education → education_substitutable
   - If lower grades allow experience-or-education AND higher grades require the degree with NO experience path → education_required_higher
   - If lower grades require experience-only AND higher grades require the degree → education_required_higher (some grade is blocked without a degree)

EXAMPLES

education_substitutable:
"To qualify for this position at the GS-11 level you must possess one of the following: One year of specialized experience equivalent to the GS-09 level... OR Ph.D. or equivalent doctoral degree OR 3 full years of progressively higher level graduate education leading to such a degree."
(Both experience AND a doctoral education path are offered. Neither is mandatory.)

no_education:
"Education is not substitutable for specialized experience at the GS-12 grade level. You must have 1 year of specialized experience equivalent to the GS-11 grade level."
(Experience is the only path — education cannot qualify you.)

education_required:
"This position has a positive education requirement in addition to the specialized experience requirement. You must possess a bachelor's degree in computer science, engineering, information science, information systems management, mathematics, operations research, statistics, or technology management."
(Degree is on top of experience, not as an alternative — the degree is mandatory.)

education_required (mandatory-degree-plus-experience pattern):
Education field: "An undergraduate degree from an accredited college/university is mandatory. The degree must be in Computer and Information Sciences and Support Services..."
Qual summary:    "GENERAL EXPERIENCE: ... SPECIALIZED EXPERIENCE: Must have at least 12 months of Cyberspace program supervisory experiences..."
(The education field says "mandatory"; the qual summary lists experience separately with no OR/in-lieu/substitute language linking them. Both are required. The "If you are using education to qualify" transcript directive is boilerplate and does not establish substitutability. → education_required.)

IMPORTANT: Your key_quote must be copied EXACTLY from the input text — do not paraphrase or summarize. If the classification is based on the ABSENCE of information (e.g., empty education field), use '(no relevant text)'."""


def build_prompt(education: str, qualification_summary: str) -> str:
    edu_text = education.strip() if education else "(empty)"
    qual_text = qualification_summary.strip() if qualification_summary else "(empty)"
    return f"""Education field:
{edu_text}

Qualifications Summary:
{qual_text}"""


def get_cache_key(education: str, qualification_summary: str) -> str:
    combined = PROMPT_VERSION + "|" + (education or "") + "|" + (qualification_summary or "")
    return hashlib.sha256(combined.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    # Billing / quota failures: retry won't help — these need human
    # intervention (top up balance, check plan tier). Fail fast.
    billing_markers = (
        "insufficient_quota",
        "insufficient quota",
        "exceeded your current quota",
        "check your plan and billing",
    )
    if any(m in s for m in billing_markers):
        return False
    return any(k in s for k in ("rate", "429", "timeout", "timed out", "connection", "overloaded"))


async def classify_one_async(
    education: str, qualification_summary: str, semaphore: asyncio.Semaphore
) -> EducationClassification:
    async with semaphore:
        user_msg = build_prompt(education, qualification_summary)
        for attempt in range(MAX_RETRIES):
            try:
                resp = await acompletion(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=EducationClassification,
                    temperature=0,
                )
                return EducationClassification.model_validate_json(
                    resp.choices[0].message.content
                )
            except Exception as e:
                if attempt == MAX_RETRIES - 1 or not _is_retryable(e):
                    raise
                # Exponential backoff with jitter, capped at 60s.
                delay = min(60, (2 ** attempt) + random.random())
                await asyncio.sleep(delay)
        # unreachable
        raise RuntimeError("retry loop exited without returning or raising")


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


async def run_classification(df: pd.DataFrame, cache: dict) -> tuple[list[dict], list[int]]:
    """Returns (results, failed_indices).

    results[i] is a dict for every successfully classified row and None for
    rows that exhausted their retries. failed_indices lists those None slots
    so the caller can drop them from the output parquet. Failed rows are
    NOT cached — a subsequent run retries them from scratch.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[dict | None] = [None] * len(df)
    failed_indices: list[int] = []
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
        return results, failed_indices

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
            failed_indices.append(idx)
            api_count += 1
            api_since_save += 1
            pbar.update(1)
            return

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
    print(f"  Final cache save ({cached_count} cached, {api_count} API calls, {len(failed_indices)} failed)")

    return results, failed_indices


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
    results, failed_indices = asyncio.run(run_classification(df, cache))

    # Drop rows that failed after all retries so we never write a row with
    # a bogus "no_education / API error" label. Re-running picks them up.
    if failed_indices:
        print(f"\n{len(failed_indices)} rows failed after {MAX_RETRIES} retries — "
              f"dropping from output. Re-run to retry them.")
        keep_mask = [i not in set(failed_indices) for i in range(len(df))]
        df = df.iloc[keep_mask].reset_index(drop=True)
        results = [r for i, r in enumerate(results) if keep_mask[i]]

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
