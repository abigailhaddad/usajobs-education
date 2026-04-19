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
import json
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_batch import (
    CONFIG, call_llm, cache_key, load_cache, save_cache, run_batch,
    submit_batch as _submit_batch, poll_batch, collect_batch,
)

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


# ── Batch API helpers ─────────────────────────────────────────────────────

BATCH_DIR = DATA_DIR / "batch"


def _make_messages(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(
            item.get("education", ""),
            item.get("qualification_summary", ""),
        )},
    ]


def _response_format_schema() -> dict:
    """JSON schema for EducationClassification that the Batch API needs.
    OpenAI structured outputs require additionalProperties: false."""
    schema = EducationClassification.model_json_schema()
    schema["additionalProperties"] = False
    # Inline $defs (OpenAI doesn't support $ref in batch mode)
    if "$defs" in schema:
        for prop in schema.get("properties", {}).values():
            if "$ref" in prop:
                ref_name = prop["$ref"].split("/")[-1]
                prop.clear()
                prop.update(schema["$defs"][ref_name])
            if "allOf" in prop:
                for item in prop["allOf"]:
                    if "$ref" in item:
                        ref_name = item["$ref"].split("/")[-1]
                        prop.clear()
                        prop.update(schema["$defs"][ref_name])
                        break
        del schema["$defs"]
    return schema


def _parse_batch_response(content: str) -> dict:
    result = EducationClassification.model_validate_json(content)
    return {
        "edu_category": result.category.value,
        "edu_reasoning": result.reasoning,
        "edu_key_quote": result.key_quote,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Classify a random sample of N jobs")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    parser.add_argument("--verify", action="store_true", help="Verify quotes in existing output")
    parser.add_argument("--batch", action="store_true",
                        help="Submit to OpenAI Batch API (50%% off, async). "
                             "Run --collect later to get results.")
    parser.add_argument("--collect", action="store_true",
                        help="Collect results from submitted batch(es) using saved metadata")
    parser.add_argument("--input", type=str,
                        help="Override input parquet (default: data/2210_raw.parquet). "
                             "When set, the 2210-specific historical merge is skipped.")
    parser.add_argument("--output", type=str,
                        help="Override output parquet path")
    parser.add_argument("--cache", type=str,
                        help="Override classification cache path")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else INPUT
    output_path = Path(args.output) if args.output else OUTPUT
    cache_path = Path(args.cache) if args.cache else CACHE_FILE
    is_custom_input = args.input is not None

    if args.verify:
        if not output_path.exists():
            print(f"No output file at {output_path}")
            return
        df = pd.read_parquet(output_path)
        bad = verify_quotes(df)
        if len(bad) == 0:
            print(f"All {len(df)} quotes verified OK")
        else:
            print(f"{len(bad)} bad quotes out of {len(df)}:")
            print(bad.to_string(index=False))
        return

    df = pd.read_parquet(input_path)
    if "data_source" not in df.columns:
        df["data_source"] = "api"
    print(f"Loaded {len(df)} jobs from {input_path.name}")
    if not is_custom_input and HISTORICAL_INPUT.exists():
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
    cache = load_cache(cache_path)

    # ── Batch API mode ────────────────────────────────────────────────
    if args.batch:
        batch_ids, uncached, cached = _submit_batch(
            items=items,
            messages_fn=_make_messages,
            cache=cache,
            cache_key_fn=make_cache_key,
            model=MODEL,
            response_format_schema=_response_format_schema(),
            batch_dir=BATCH_DIR,
        )
        if batch_ids:
            print(f"\n{len(batch_ids)} batch(es) submitted:")
            for bid in batch_ids:
                print(f"  {bid}")
            print(f"\nCollect results when done:")
            print(f"  python classify.py --collect")
        else:
            print("\nAll items cached — nothing to submit.")
        return

    if args.collect:
        meta = json.loads((BATCH_DIR / "batch_meta.json").read_text())
        batch_ids = meta.get("batch_ids", [meta["batch_id"]] if "batch_id" in meta else [])
        cached_set = set(meta["uncached_indices"])
        cached_indices = [i for i in range(len(items)) if i not in cached_set]

        # Collect all chunks
        results: list[dict | None] = [None] * len(items)
        all_failed: list[int] = []

        # Fill cached
        for i in cached_indices:
            key = make_cache_key(items[i])
            if key in cache:
                results[i] = cache[key]

        for ci, chunk_meta in enumerate(meta.get("chunks", [meta])):
            bid = chunk_meta["batch_id"]
            print(f"Collecting chunk {ci+1}/{len(meta.get('chunks', [meta]))}: {bid}")
            batch_result = poll_batch(bid)
            chunk_results, chunk_failed = collect_batch(
                batch_result=batch_result,
                items=items,
                uncached_indices=chunk_meta["uncached_indices"],
                cached_indices=[],  # cached already filled above
                cache=cache,
                cache_key_fn=make_cache_key,
                cache_path=cache_path,
                parse_fn=_parse_batch_response,
            )
            # Merge chunk results into main results
            for i in chunk_meta["uncached_indices"]:
                if chunk_results[i] is not None:
                    results[i] = chunk_results[i]
            all_failed.extend(chunk_failed)

        # Fill any remaining Nones from cache (rows cached between
        # submission and collection, e.g. from test runs)
        backfilled = 0
        for i in range(len(results)):
            if results[i] is None:
                key = make_cache_key(items[i])
                if key in cache:
                    results[i] = cache[key]
                    backfilled += 1
        if backfilled:
            print(f"  Backfilled {backfilled} rows from cache")

        # Any still-None are real failures
        all_failed = [i for i in range(len(results)) if results[i] is None]
        _write_output(df, results, all_failed, output_path)
        return

    # ── Real-time mode ────────────────────────────────────────────────
    results, failed = asyncio.run(run_batch(
        items=items,
        process_fn=classify_one,
        cache=cache,
        cache_key_fn=make_cache_key,
        cache_path=cache_path,
        desc="Classifying",
    ))
    _write_output(df, results, failed, output_path)


def _write_output(df, results, failed, output_path=OUTPUT):
    if failed:
        print(f"\n{len(failed)} rows failed — dropping from output. Re-run to retry.")
        keep = [i not in set(failed) for i in range(len(df))]
        df = df.iloc[keep].reset_index(drop=True)
        results = [r for i, r in enumerate(results) if keep[i]]

    result_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    out.to_parquet(output_path, index=False)
    print(f"\nWrote {len(out)} classified jobs to {output_path}")
    print(f"\n{out['edu_category'].value_counts().to_string()}")

    bad = verify_quotes(out)
    if len(bad) == 0:
        print(f"\nAll quotes verified OK")
    else:
        print(f"\n{len(bad)} bad quotes (not found in source text):")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
