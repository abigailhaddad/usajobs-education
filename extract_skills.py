#!/usr/bin/env python3
"""
Extract skills/experience requirements from 2210 IT job postings using an LLM.

Reads the raw parquet, sends education + qualification_summary to the model,
and gets back structured skills data (overall + per-grade).

Usage:
    python extract_skills.py                # Extract all
    python extract_skills.py --sample 10    # Test on 10
"""

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_verified.parquet"
OUTPUT = DATA_DIR / "2210_skills.json"
CACHE_FILE = DATA_DIR / "skills_cache.json"

MODEL = "openai/gpt-5.4-mini"
MAX_CONCURRENT = 5
MAX_RETRIES = 5


class GradeSkills(BaseModel):
    grade: str = Field(description="Grade level, e.g. 'GS-09', 'GS-11', 'GG-13'. Use 'all' if the posting doesn't distinguish by grade.")
    specialized_experience: list[str] = Field(description="Specific skills, tasks, or experience items required at this grade. Each item should be a concise phrase (not a full sentence). E.g. 'configuring network routers and switches', 'vulnerability scanning with Nessus or similar tools'")


class SkillsExtraction(BaseModel):
    specialization: str = Field(description="The IT specialization area, e.g. 'INFOSEC', 'NETWORK', 'SYSADMIN', 'APPSW', 'CUSTSPT', 'DATAMGT', 'PLCYPLN', 'GENERAL'. Infer from the title parenthetical or the duties described.")
    technical_skills: list[str] = Field(description="Specific technologies, tools, platforms, certifications, or protocols mentioned. E.g. 'AWS', 'CompTIA Security+', 'Active Directory', 'Python', 'Splunk'. Only include explicitly named technologies, not generic descriptions.")
    skill_areas: list[str] = Field(description="Broader skill domains required. E.g. 'network administration', 'cybersecurity', 'cloud migration', 'database management', 'project management'. Keep to 3-7 items.")
    by_grade: list[GradeSkills] = Field(description="Skills broken out by grade level. If the posting only lists one set of requirements, use grade='all'.")
    certifications_mentioned: list[str] = Field(description="Any certifications explicitly named (e.g. 'CISSP', 'Security+', 'PMP', 'CCNA'). Empty list if none.")
    clearance_level: str = Field(description="Security clearance required, e.g. 'Top Secret/SCI', 'Secret', 'None', 'Not specified'")


SYSTEM_PROMPT = """You are extracting structured skills and experience requirements from federal 2210 IT Specialist job postings.

You will receive the position title, education field, and qualifications summary. Extract:

1. **specialization**: The IT sub-field. Look at the title parenthetical (e.g. "INFOSEC", "NETWORK", "CUSTSPT") or infer from duties. Use one of: INFOSEC, NETWORK, SYSADMIN, APPSW, CUSTSPT, DATAMGT, PLCYPLN, GENERAL, or a brief custom label if none fit.

2. **technical_skills**: Explicitly named technologies, tools, platforms, protocols, or products. Only extract things that are specifically named — not generic descriptions like "network tools." Examples: "Splunk", "AWS GovCloud", "Cisco IOS", "ServiceNow", "Python".

3. **skill_areas**: Broader skill domains (3-7 items). E.g. "network security monitoring", "incident response", "cloud infrastructure management".

4. **by_grade**: Skills/experience broken out by grade level. Many postings define different specialized experience for GS-09 vs GS-11 vs GS-12. Extract the specific tasks/skills for each grade mentioned. If the posting doesn't distinguish, use grade="all".

5. **certifications_mentioned**: Any certs explicitly named. Don't infer — only extract if the text says the cert name.

6. **clearance_level**: The security clearance mentioned.

IMPORTANT:
- Extract what the posting ACTUALLY says, not what you think the job might need.
- For specialized_experience items, be concise: "managing Active Directory" not "Experience in managing and maintaining Active Directory environments including user provisioning and group policy management."
- Ignore the four OPM IT competencies boilerplate (Attention to Detail, Customer Service, Oral Communication, Problem Solving) — these appear on every 2210 posting and aren't useful.
- Ignore generic statements like "must have 1 year of specialized experience at the next lower grade level" — extract WHAT the experience is in, not the time requirement."""


def build_prompt(row: pd.Series) -> str:
    title = row["position_title"] if pd.notna(row["position_title"]) else ""
    qual = str(row["qualification_summary"]).strip() if pd.notna(row["qualification_summary"]) else "(empty)"
    return f"""Position Title: {title}

Qualifications / Specialized Experience:
{qual}"""


def get_cache_key(row: pd.Series) -> str:
    combined = (
        (str(row["position_title"]) if pd.notna(row["position_title"]) else "")
        + "|" + (str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else "")
    )
    return hashlib.sha256(combined.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


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


async def extract_one(row: pd.Series, semaphore: asyncio.Semaphore) -> SkillsExtraction:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await acompletion(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(row)},
                    ],
                    response_format=SkillsExtraction,
                    temperature=0,
                )
                return SkillsExtraction.model_validate_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt == MAX_RETRIES - 1 or not _is_retryable(e):
                    raise
                delay = min(60, (2 ** attempt) + random.random())
                await asyncio.sleep(delay)
        raise RuntimeError("retry loop exited without returning or raising")


async def run_extraction(df: pd.DataFrame, cache: dict) -> tuple[list[dict | None], list[int]]:
    """Returns (results, failed_indices).

    Failed rows are NOT cached and their slot in results stays None.
    Caller drops them from the output JSON so a re-run retries cleanly.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[dict | None] = [None] * len(df)
    failed_indices: list[int] = []
    cached_count = 0
    uncached_indices = []

    rows_list = list(df.iterrows())
    for i, (_, row) in enumerate(rows_list):
        key = get_cache_key(row)
        if key in cache:
            results[i] = cache[key]
            cached_count += 1
        else:
            uncached_indices.append(i)

    print(f"  {cached_count} cached, {len(uncached_indices)} need API calls")

    if not uncached_indices:
        return results, failed_indices

    pbar = tqdm(total=len(uncached_indices), desc="Extracting skills", unit="job")
    api_since_save = 0

    async def process_one(idx):
        nonlocal api_since_save
        _, row = rows_list[idx]
        try:
            result = await extract_one(row, semaphore)
            result_dict = result.model_dump()
        except Exception as e:
            tqdm.write(f"  ERROR {row['usajobs_control_number']}: {e}")
            failed_indices.append(idx)
            api_since_save += 1
            pbar.update(1)
            return

        key = get_cache_key(row)
        cache[key] = result_dict
        results[idx] = result_dict
        api_since_save += 1
        pbar.update(1)

        if api_since_save >= 100:
            save_cache(cache)
            api_since_save = 0
            tqdm.write(f"  Cache saved")

    batch_size = MAX_CONCURRENT * 2
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch = uncached_indices[batch_start : batch_start + batch_size]
        await asyncio.gather(*[process_one(idx) for idx in batch])

    pbar.close()
    save_cache(cache)
    return results, failed_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Extract from a random sample")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    # Exclude not_a_posting
    df = df[df["edu_category"] != "not_a_posting"].reset_index(drop=True)
    print(f"Loaded {len(df)} jobs")

    if args.sample:
        df = df.sample(args.sample, random_state=42)
        print(f"Sampled {len(df)} jobs")

    cache = load_cache()
    results, failed_indices = asyncio.run(run_extraction(df, cache))

    if failed_indices:
        print(f"\n{len(failed_indices)} extractions failed after {MAX_RETRIES} "
              f"retries — dropping from output. Re-run to retry them.")

    # Combine with job metadata — skip rows where extraction failed.
    failed_set = set(failed_indices)
    output = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i in failed_set or results[i] is None:
            continue
        entry = {
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
        }
        output.append(entry)

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(output)} jobs to {OUTPUT}")

    # Quick stats
    all_tech = []
    all_certs = []
    all_specs = []
    for r in results:
        all_tech.extend(r["technical_skills"])
        all_certs.extend(r["certifications_mentioned"])
        all_specs.append(r["specialization"])

    from collections import Counter
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
