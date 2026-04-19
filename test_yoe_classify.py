#!/usr/bin/env python3
"""
Validation test for yoe_classify.regex_classify against hand-labeled examples.

Multi-label: each test case has a SET of expected YoeCategory values.

Usage:
    python test_yoe_classify.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from yoe_classify import YoeCategory, regex_classify

DATA_DIR = Path(__file__).parent / "data"

# Hand-labeled test cases: (control_number, expected_set, reason)
TEST_CASES: list[tuple[str, set[YoeCategory], str]] = [
    # ── yoe_multi_year ────────────────────────────────────────────────────
    ("832470300", {YoeCategory.multi_year},
     "'36 months' specialized experience — GS-11 posting"),
    ("746584800", {YoeCategory.multi_year},
     "'24 months' specialized experience — GS-9 posting"),
    ("786569000", {YoeCategory.multi_year},
     "NF-5 posting, 'five years of experience'"),

    # ── yoe_one_year (standard federal pattern) ───────────────────────────
    ("792583800", {YoeCategory.one_year},
     "GS-13 posting, 'one year of specialized experience at GS-12'. "
     "4-competency 'general experience' is the basic requirement, not a separate no-exp path"),
    ("773575800", {YoeCategory.one_year},
     "GS-9-11 education-or-experience, experience path is 1 year at prior grade"),
    ("708255500", {YoeCategory.one_year},
     "GS-14, 'at least one year of experience equal to GS-13'"),
    ("767039100", {YoeCategory.one_year},
     "GS-5-12 multi-grade, each grade requires 1 year at next lower"),
    ("746099600", {YoeCategory.one_year},
     "GS-14, 'at least one (1) year of specialized experience at the next lower grade GS-13'"),
    ("780663300", {YoeCategory.one_year},
     "GS-11, 'One year of specialized experience'"),

    # ── collapse rule: mixed no-exp + 1yr grades collapse to yoe_one_year ─
    ("747537500", {YoeCategory.one_year},
     "FV-H/I. FV-G has no time and FV-H has 1 year at FV-G. Collapsed: yoe_one_year only."),
    ("702652100", {YoeCategory.one_year},
     "GS-5-7. GS-5 has only 4 competencies (no time) but GS-7 has 1 year. Collapsed: yoe_one_year."),

    # ── pure yoe_no_experience ────────────────────────────────────────────
    ("706400000", {YoeCategory.no_experience},
     "GS-5 only, experience path requires only 4 IT competencies — no time threshold"),
    ("697782100", {YoeCategory.no_experience},
     "GG-14 DCIPS. SPECIALIZED EXPERIENCE section is qualitative ('quality level of experience') "
     "with no time threshold."),
    ("730795400", {YoeCategory.no_experience},
     "GS-12 INFOSEC. Experience requirement is qualitative duties list with no time threshold."),
    ("821261200", {YoeCategory.no_experience},
     "GG-13 POLICY. 'Applicant must have directly applicable experience' — qualitative, no time."),
]


def main():
    df = pd.read_parquet(DATA_DIR / "2210_classified.parquet")
    df["cn"] = df["usajobs_control_number"].astype(str)
    df = df.set_index("cn")

    correct = 0
    wrong = []
    errors = []
    predictions: list[tuple[str, set[str], set[str]]] = []

    for cn, expected, reason in TEST_CASES:
        if cn not in df.index:
            errors.append((cn, "NOT FOUND in dataset"))
            continue

        row = df.loc[cn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        result = regex_classify(row.to_dict())
        predicted = set(result["yoe_categories"])
        expected_strs = {c.value for c in expected}
        predictions.append((cn, expected_strs, predicted))

        if predicted == expected_strs:
            correct += 1
            print(f"  ✓ {cn}  {sorted(expected_strs)}")
        else:
            wrong.append((cn, expected_strs, predicted, result["yoe_reasoning"], reason))
            print(f"  ✗ {cn}  expected={sorted(expected_strs)}  got={sorted(predicted)}")
            print(f"         reason: {result['yoe_reasoning'][:160]}")

    n_tested = len(TEST_CASES) - len(errors)
    print(f"\n{'='*60}")
    print(f"Exact-match accuracy: {correct}/{n_tested} ({100*correct/n_tested:.0f}%)")

    if wrong:
        print(f"\n{len(wrong)} WRONG:")
        for cn, exp, got, reasoning, reason in wrong:
            print(f"  {cn}: expected {sorted(exp)}, got {sorted(got)}")
            print(f"    label reason: {reason}")
            print(f"    classifier:   {reasoning[:200]}")
            print()
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for cn, msg in errors:
            print(f"  {cn}: {msg}")

    print("\nPer-category (multi-label):")
    for cat in YoeCategory:
        tp = sum(1 for _, e, p in predictions if cat.value in e and cat.value in p)
        fp = sum(1 for _, e, p in predictions if cat.value not in e and cat.value in p)
        fn = sum(1 for _, e, p in predictions if cat.value in e and cat.value not in p)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        print(f"  {cat.value:22s}  tp={tp} fp={fp} fn={fn}  precision={prec:.2f} recall={rec:.2f}")


if __name__ == "__main__":
    main()
