#!/usr/bin/env python3
"""
Validation test for classify.py against hand-labeled examples.

Each test case has a control number and the expected EducationCategory.
Runs the classifier, compares predicted vs expected, reports accuracy
per category.

Usage:
    python test_classify.py
"""

import asyncio
from pathlib import Path

import pandas as pd

from classify import EducationCategory, classify_one, make_cache_key
from llm_batch import load_cache, save_cache

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "classification_cache.json"

# Hand-labeled test cases: (control_number, expected_category, reason)
# These were manually verified by reading the posting text.
TEST_CASES: list[tuple[str, EducationCategory, str]] = [
    # ── no_education ──────────────────────────────────────────────────────
    # Experience-only, education explicitly blocked
    ("814836500", EducationCategory.no_education,
     "edu says 'you must meet the qualification requirement using experience alone'"),
    ("839248400", EducationCategory.no_education,
     "edu says 'Education is not substitutable for specialized experience at this grade level'"),
    ("839262500", EducationCategory.no_education,
     "edu says 'no substitution of education for the qualifying experience at the GG-14 grade level'"),
    ("839318700", EducationCategory.no_education,
     "edu says 'no substitution of education for experience at this grade level'"),
    ("838125400", EducationCategory.no_education,
     "edu says 'you must meet the qualification requirement using experience alone'"),
    # Empty education field, experience-only qual
    ("807382700", EducationCategory.no_education,
     "edu is empty, qual requires specialized experience only at DS-04"),
    ("817612400", EducationCategory.no_education,
     "edu is empty, qual requires 1yr IT experience at GS-12"),
    ("827501800", EducationCategory.no_education,
     "edu is empty, qual requires specialized experience at GS-11"),
    ("832738900", EducationCategory.no_education,
     "edu is empty, qual requires cybersecurity experience at GS-11"),

    # ── education_substitutable ───────────────────────────────────────────
    # OPM Alternative A: experience OR degree
    ("810088100", EducationCategory.education_substitutable,
     "edu lists 'Major study--computer science...' with substitution language"),
    ("838617700", EducationCategory.education_substitutable,
     "edu says 'Ph.D. or equivalent, or 3 years of graduate education' at GS-11"),
    ("842383100", EducationCategory.education_substitutable,
     "edu says 'Substitution of Education for Specialized Experience' at GS-5/7/9"),
    ("844723600", EducationCategory.education_substitutable,
     "edu says 'SUBSTITUTION OF EDUCATION FOR SPECIALIZED EXPERIENCE'"),
    # Empty edu but OR-education path in qual summary
    ("806611100", EducationCategory.education_substitutable,
     "edu is empty but qual says 'EXPERIENCE OR EDUCATION as described below'"),
    # Military OR-training/certification = degree not mandatory
    ("842708200", EducationCategory.education_substitutable,
     "edu lists 'Bachelors in IT OR DoD/Military Training OR Certification'"),
    ("842708400", EducationCategory.education_substitutable,
     "edu lists 'Bachelors in IT OR DoD/Military Training OR Certification'"),
    # Education works at some grades but blocked at higher
    ("736214600", EducationCategory.education_substitutable,
     "edu at GS-9/11 but 'no substitution' at GS-12/13 — edu IS a path at lower grades"),

    # ── education_required ────────────────────────────────────────────────
    # Degree explicitly mandatory, no OR/in-lieu language
    ("839859600", EducationCategory.education_required,
     "edu says 'An undergraduate degree from an accredited college/university is mandatory'"),
    ("831587200", EducationCategory.education_required,
     "edu says 'must possess a bachelor's degree' with no experience alternative"),
    ("841690000", EducationCategory.education_required,
     "edu says 'bachelor's degree in IT... and 3 years of experience are required'"),
    ("749843600", EducationCategory.education_required,
     "edu says 'Mandatory Requirements: Candidates must have a degree in computer science...'"),

    # ── not_a_posting ─────────────────────────────────────────────────────
    # DHA notice / placeholder (need to find real examples)
    ("811434300", EducationCategory.not_a_posting,
     "edu says 'this notice is not posted for applications'"),
]


async def main():
    # Load all test-case rows from the combined dataset
    df = pd.read_parquet(DATA_DIR / "2210_raw.parquet")
    hist_path = DATA_DIR / "2210_historical_raw.parquet"
    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        df = pd.concat([df, hist], ignore_index=True)
    df["cn"] = df["usajobs_control_number"].astype(str)
    df = df.set_index("cn")

    cache = load_cache(CACHE_FILE)
    sem = asyncio.Semaphore(5)

    # Run classifier on each test case
    print(f"Running {len(TEST_CASES)} test cases\n")

    correct = 0
    wrong = []
    errors = []

    for cn, expected, reason in TEST_CASES:
        if cn not in df.index:
            errors.append((cn, expected, "NOT FOUND in dataset"))
            continue

        row = df.loc[cn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        item = row.to_dict()

        try:
            result = await classify_one(item, sem)
        except Exception as e:
            errors.append((cn, expected, f"API error: {e}"))
            continue

        predicted = result["edu_category"]
        key = make_cache_key(item)
        cache[key] = result

        if predicted == expected.value:
            correct += 1
            print(f"  ✓ {cn}  {expected.value}")
        else:
            wrong.append((cn, expected.value, predicted, result["edu_reasoning"], reason))
            print(f"  ✗ {cn}  expected={expected.value}  got={predicted}")
            print(f"         model: {result['edu_reasoning'][:150]}")

    save_cache(cache, CACHE_FILE)

    # Summary
    total = len(TEST_CASES)
    n_err = len(errors)
    n_tested = total - n_err
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{n_tested} correct ({100*correct/n_tested:.0f}%)")
    if wrong:
        print(f"\n{len(wrong)} WRONG:")
        for cn, exp, got, reasoning, reason in wrong:
            print(f"  {cn}: expected {exp}, got {got}")
            print(f"    hand-label reason: {reason}")
            print(f"    model reasoning:   {reasoning[:200]}")
            print()
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for cn, exp, msg in errors:
            print(f"  {cn}: {msg}")

    # Per-category breakdown
    print(f"\nPer-category:")
    from collections import Counter
    expected_counts = Counter(e.value for _, e, _ in TEST_CASES)
    correct_by_cat: dict[str, int] = {}
    for cn, expected, reason in TEST_CASES:
        if cn not in df.index:
            continue
        row = df.loc[cn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        key = make_cache_key(row.to_dict())
        if key in cache:
            if cache[key]["edu_category"] == expected.value:
                correct_by_cat[expected.value] = correct_by_cat.get(expected.value, 0) + 1
    for cat in EducationCategory:
        n = expected_counts.get(cat.value, 0)
        c = correct_by_cat.get(cat.value, 0)
        if n:
            print(f"  {cat.value:30s}  {c}/{n} correct")


if __name__ == "__main__":
    asyncio.run(main())
