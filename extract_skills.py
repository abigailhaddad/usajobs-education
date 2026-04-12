#!/usr/bin/env python3
"""
Extract skills/experience requirements from 2210 IT job postings.

Usage:
    python extract_skills.py                # Extract all
    python extract_skills.py --sample 10    # Test on 10
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_batch import CONFIG, call_llm, cache_key, load_cache, save_cache, run_batch

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_verified.parquet"
OUTPUT = DATA_DIR / "2210_skills.json"
CACHE_FILE = DATA_DIR / "skills_cache.json"

TASK = CONFIG["extract_skills"]
MODEL = TASK["model"]
SYSTEM_PROMPT = TASK["system_prompt"]


# ── Schema ────────────────────────────────────────────────────────────────

class GradeSkills(BaseModel):
    grade: str = Field(
        description="Grade level, e.g. 'GS-09', 'GS-11', 'GG-13'. "
                    "Use 'all' if the posting doesn't distinguish by grade."
    )
    specialized_experience: list[str] = Field(
        description="Specific skills, tasks, or experience items required at this grade."
    )


class SkillsExtraction(BaseModel):
    specialization: str = Field(
        description="The IT specialization area, e.g. 'INFOSEC', 'NETWORK', 'SYSADMIN', "
                    "'APPSW', 'CUSTSPT', 'DATAMGT', 'PLCYPLN', 'GENERAL'."
    )
    technical_skills: list[str] = Field(
        description="Specific technologies, tools, platforms, certifications, or protocols mentioned."
    )
    skill_areas: list[str] = Field(
        description="Broader skill domains required (3-7 items)."
    )
    by_grade: list[GradeSkills] = Field(
        description="Skills broken out by grade level."
    )
    certifications_mentioned: list[str] = Field(
        description="Any certifications explicitly named. Empty list if none."
    )
    clearance_level: str = Field(
        description="Security clearance required, e.g. 'Top Secret/SCI', 'Secret', 'None'."
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    title = item.get("position_title") or ""
    qual = str(item.get("qualification_summary") or "").strip() or "(empty)"
    return f"Position Title: {title}\n\nQualifications / Specialized Experience:\n{qual}"


def make_cache_key(item: dict) -> str:
    return cache_key(
        item.get("position_title") or "",
        item.get("qualification_summary") or "",
    )


async def extract_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        result = await call_llm(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(item)},
            ],
            response_format=SkillsExtraction,
        )
        return result.model_dump()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Extract from a random sample")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    df = df[df["edu_category"] != "not_a_posting"].reset_index(drop=True)
    print(f"Loaded {len(df)} jobs")

    if args.sample:
        df = df.sample(args.sample, random_state=42)
        print(f"Sampled {len(df)} jobs")

    items = df.to_dict(orient="records")
    cache = load_cache(CACHE_FILE)
    results, failed = asyncio.run(run_batch(
        items=items,
        process_fn=extract_one,
        cache=cache,
        cache_key_fn=make_cache_key,
        cache_path=CACHE_FILE,
        desc="Extracting skills",
    ))

    if failed:
        print(f"\n{len(failed)} extractions failed — dropping from output. Re-run to retry.")

    failed_set = set(failed)
    output = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i in failed_set or results[i] is None:
            continue
        output.append({
            "usajobs_control_number": row["usajobs_control_number"],
            "usajobs_url": row["usajobs_url"],
            "position_title": row["position_title"],
            "agency": row["agency"],
            "department": row["department"],
            "min_grade": row["min_grade"],
            "max_grade": row["max_grade"],
            "source_year": int(row["source_year"]),
            "edu_category": row["edu_category"],
            "data_source": row.get("data_source", "api"),
            **results[i],
        })

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(output)} jobs to {OUTPUT}")

    all_tech = []
    all_certs = []
    all_specs = []
    for r in results:
        if r is None:
            continue
        all_tech.extend(r["technical_skills"])
        all_certs.extend(r["certifications_mentioned"])
        all_specs.append(r["specialization"])

    print(f"\nTop specializations:")
    for s, c in Counter(all_specs).most_common(10):
        print(f"  {s}: {c}")
    print(f"\nTop technical skills:")
    for s, c in Counter(all_tech).most_common(15):
        print(f"  {s}: {c}")
    print(f"\nTop certifications:")
    for s, c in Counter(all_certs).most_common(10):
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
