#!/usr/bin/env python3
"""
Classify years-of-experience requirements for 2210 IT jobs.

Pure regex classifier — fast, deterministic, no LLM calls. The collapsed
taxonomy (yoe_no_experience mutually exclusive with yoe_one_year and
yoe_multi_year) is simple enough that regex handles it cleanly.

Categories:
- yoe_no_experience: posting describes experience qualitatively with NO time
  threshold anywhere (DCIPS / IC / GG-series, "quality level of experience",
  KSA bullets, or 4-IT-competencies-only entry tier)
- yoe_one_year: at least one advertised grade requires ~1 year specialized
  experience at the next-lower grade
- yoe_multi_year: at least one advertised grade requires >1 year (24+ months,
  2+ years, etc.) — typical of Title-32, NF/NH pay scales, senior roles
- yoe_ses_or_executive: SES postings (regex-routed via service_type)

Collapse rule: if any advertised grade has a time threshold, we don't add
yoe_no_experience even when other grades have only qualitative text. This
keeps the taxonomy mutually exclusive between no_experience and the time-
threshold buckets, while letting yoe_one_year and yoe_multi_year co-occur
when grades within the same posting have different thresholds.

Usage:
    python yoe_classify.py             # Classify all eligible rows
    python yoe_classify.py --sample 100  # Test on 100 random rows
"""

import argparse
import re
from collections import Counter
from enum import Enum
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
INPUT = DATA_DIR / "2210_classified.parquet"
OUTPUT = DATA_DIR / "2210_yoe.parquet"

# Every row gets a YOE label — education_required rows still have experience
# requirements on top of the mandatory degree, and not_a_posting rows are
# regex-routed to the yoe_not_a_posting bucket so they're visible with an
# explicit tag instead of being silently skipped.
ELIGIBLE_EDU = {"no_education", "education_substitutable", "education_required"}


class YoeCategory(str, Enum):
    no_experience = "yoe_no_experience"
    one_year = "yoe_one_year"
    multi_year = "yoe_multi_year"
    # yoe_ses_or_executive and yoe_not_a_posting are regex-routed via
    # SES_SHORTCUT / NOT_A_POSTING_SHORTCUT — never produced by the
    # experience classifier itself.


# SES rows use ECQs/TQs rather than the GS-ladder "1 year at prior grade"
# taxonomy, so they're routed out before regex classification.
SES_CATEGORY = "yoe_ses_or_executive"
SES_SHORTCUT = {
    "yoe_categories": [SES_CATEGORY],
    "yoe_reasoning": "SES posting (service_type='Senior Executive'). "
                     "Uses ECQs/TQs rather than GS-ladder experience requirements — "
                     "regex-routed without any classifier call.",
    "yoe_key_quote": "(SES shortcut — service_type='Senior Executive')",
    "yoe_quotes": {},
}

NOT_A_POSTING_CATEGORY = "yoe_not_a_posting"
NOT_A_POSTING_SHORTCUT = {
    "yoe_categories": [NOT_A_POSTING_CATEGORY],
    "yoe_reasoning": "Tagged not_a_posting by the education classifier "
                     "(DHA notice / placeholder / not posted for applications). "
                     "No experience classification applied.",
    "yoe_key_quote": "(not a posting — see edu classification)",
    "yoe_quotes": {},
}


def is_ses(item: dict) -> bool:
    st = item.get("service_type")
    return isinstance(st, str) and st.strip() == "Senior Executive"


def is_not_a_posting(item: dict) -> bool:
    return item.get("edu_category") == "not_a_posting"


# ── Regex patterns ───────────────────────────────────────────────────────
# Grade prefixes the federal pay scales actually use in postings.
_GRADE_PREFIXES = (
    "GS|GG|FV|NH|NT|ZP|NF|DS|CG|CT|DE|IC|IA|IP|IR|IS|IT|"
    "LE|NM|NO|NY|SV|TR|WG|WL|WS|ZA|ZS|AD"
)

# "(at least) (one|1) year ... experience" — bounded by [^.] to stay in one
# sentence. Matches "one year of specialized experience", "one year of IT-
# related experience", "one year of experience, education, or training", etc.
# Also "one year ... at the next lower grade" (handles "or will have one year
# within N days of closing, at the next lower grade").
_ONE_YEAR_PATTERNS = [
    r"\b(?:at\s+least\s+)?(?:one|1)[-\s]*(?:\(1\)[-\s]*)?year[^.]{0,80}?\bexperience",
    r"\b52\s*weeks?\s+of(?:\s+\w+){0,5}?\s+experience",
    rf"\b52\s*weeks?\s+of(?:\s+\w+){{0,5}}?\s+at\s+(?:the\s+)?(?:next\s+(?:lower|higher)?\s*(?:grade|pay[-\s]?band|level)|equivalent\s+grade|equivalent\s+to\s+(?:the\s+)?(?:{_GRADE_PREFIXES})[-\s]?\d)",
    # TIG (time-in-grade) language — functionally equivalent to 1 year at prior grade.
    r"\btime[-\s]?in[-\s]?grade[^.]{0,150}?52\s*weeks?",
    r"\bTIG[^.]{0,80}?52\s*weeks?",
    r"\b52\s*weeks?\s+(?:at|of\s+service\s+at)\s+(?:the\s+)?next\s+(?:lower|higher)?\s*(?:grade|level|pay[-\s]?band)",
    r"\b12\s+months\s+of(?:\s+\w+){0,5}?\s+experience",
    r"\b12\s+months\s+experience",
    rf"\b(?:at\s+least\s+)?12\s+months[^.]{{0,300}}?(?:next\s+lower\s+(?:grade|pay[-\s]?band|level)|equivalent\s+to\s+(?:the\s+)?(?:{_GRADE_PREFIXES})[-\s]?\d)",
    rf"\b(?:one|1)[-\s]*(?:\(1\)[-\s]*)?year[^.]{{0,300}}?(?:next\s+lower\s+(?:grade|pay[-\s]?band|level)|equivalent\s+to\s+(?:the\s+)?(?:{_GRADE_PREFIXES})[-\s]?\d)",
    rf"\bat\s+least\s+(?:one|1)[-\s]*(?:\(1\)[-\s]*)?year[^.]{{0,300}}?(?:next\s+lower\s+(?:grade|pay[-\s]?band|level)|equivalent\s+to\s+(?:the\s+)?(?:{_GRADE_PREFIXES})[-\s]?\d)",
]

# >1 year explicit thresholds. Excludes "1 year" (use [2-9]|[1-9]\d+).
# "X years of [up to ~5 words] experience" / "24+ months ... experience".
_MULTI_YEAR_PATTERNS = [
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:\(\d+\)\s+)?(?:or\s+more\s+)?years?['’]?(?:\s+of)?(?:\s+\w+){0,5}?\s+experience",
    r"\b(?:[2-9]|[1-9]\d+)\s+(?:or\s+more\s+)?years?['’]?(?:\s+of)?(?:\s+\w+){0,5}?\s+experience",
    r"\bat\s+least\s+(?:two|three|four|five|six|seven|eight|nine|ten|[2-9]|[1-9]\d+)\s+(?:\(\d+\)\s+)?years?",
    r"\b(?:24|30|36|48|60|72|84|96|108|120)\s+months\b[^.]{0,30}?(?:specialized|IT|experience|progressively|\bin\s+)",
    r"\b(?:minimum|at\s+least)\s+(?:of\s+)?(?:[2-9]|[1-9]\d+|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+of",
]

_ONE_YEAR_RE = re.compile("|".join(_ONE_YEAR_PATTERNS), re.IGNORECASE)
_MULTI_YEAR_RE = re.compile("|".join(_MULTI_YEAR_PATTERNS), re.IGNORECASE)

# False-positive guard: "X years old" / "X years of age" — age requirements
# (common in NF/NF-equivalent postings) must not trigger multi_year.
_AGE_EXCLUDE_RE = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:\(\d+\)\s+)?years?\s+(?:old|of\s+age)",
    re.IGNORECASE,
)



# ── Preferred / desired experience is not required experience ────────────
# A years-of-experience span that sits in a clause saying the experience is
# only preferred, desired, a plus, or not required does not set the posting's
# requirement. Clauses are split on sentence ends, ';', newlines and bullets.
_CLAUSE_SPLIT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)|;|\n|[\u2022\u25cf\u25aa\u00b7\u2023]")
_PREFERRED_RE = re.compile(
    r"\bprefer(?:red|s)?\b|\bdesired\b|\ba\s+plus\b|\bnice\s+to\s+have\b"
    r"|\bnot\s+(?:be\s+)?(?:required|necessary)\b|\bis\s+optional\b|\bhelpful\b",
    re.IGNORECASE,
)
# Negated forms are removed before looking for requirement words, so that
# "preferred but not required" does not count as "required".
_NEGATED_REQ_RE = re.compile(r"\bnot\s+(?:be\s+)?(?:required|necessary)\b", re.IGNORECASE)
_REQUIRED_RE = re.compile(r"\brequired\b|\bmust\b|\bminimum\b", re.IGNORECASE)
# "preferably in X" and "(... sectors preferred)" qualify the experience, they
# do not make the years optional: drop parentheticals (except a bare
# "(preferred)" / "(desired)") before looking for preference words.
_PAREN_RE = re.compile(r"\([^)]*\)")
_BARE_PREF_PAREN_RE = re.compile(
    r"\(\s*(?:highly\s+)?(?:preferred|desired|a\s+plus|nice\s+to\s+have)\s*\)", re.IGNORECASE)


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Span of the clause(s) touched by text[start:end]."""
    lo = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text, 0, start):
        lo = m.end()
    m = _CLAUSE_SPLIT_RE.search(text, end)
    hi = m.start() if m else len(text)
    return lo, hi


def _is_preferred_only(text: str, start: int, end: int) -> bool:
    lo, hi = _clause_bounds(text, start, end)
    clause = text[lo:hi]
    pref_view = _PAREN_RE.sub(lambda m: m.group(0) if _BARE_PREF_PAREN_RE.fullmatch(m.group(0)) else " ", clause)
    if not _PREFERRED_RE.search(pref_view):
        return False
    return not _REQUIRED_RE.search(_NEGATED_REQ_RE.sub(" ", clause))


def _first_required_match(regex: re.Pattern, text: str):
    """First regex match that is not in a preferred/desired-only clause."""
    for m in regex.finditer(text):
        if not _is_preferred_only(text, m.start(), m.end()):
            return m
    return None


def regex_classify(item: dict) -> dict:
    """Classify a row by regex. Returns a dict ready to attach to the row.

    yoe_quotes: dict keyed by category, mapping to the verbatim span from the
    source text that triggered that category's match. Lets the site show a
    different justifying quote per badge in multi-label rows.
    yoe_key_quote: kept for backward compat — the most specific quote
    (multi_year > one_year > fallback).
    """
    text = (item.get("education") or "") + "\n" + (item.get("qualification_summary") or "")
    # Same-length substitution preserves match indices for quote extraction.
    cleaned = _AGE_EXCLUDE_RE.sub(lambda m: " " * len(m.group(0)), text)

    one_m = _first_required_match(_ONE_YEAR_RE, cleaned)
    multi_m = _first_required_match(_MULTI_YEAR_RE, cleaned)

    cats: list[str] = []
    quotes: dict[str, str] = {}
    if multi_m:
        cats.append(YoeCategory.multi_year.value)
        quotes[YoeCategory.multi_year.value] = text[multi_m.start():multi_m.end()].strip()
    if one_m:
        cats.append(YoeCategory.one_year.value)
        quotes[YoeCategory.one_year.value] = text[one_m.start():one_m.end()].strip()

    if not cats:
        cats = [YoeCategory.no_experience.value]
        reasoning = ("Regex: no 'X year' / 'X month' / '52 weeks' specialized-"
                     "experience time-threshold language found in the posting text.")
        key_quote = "(no relevant text)"
    else:
        reasoning = f"Regex-matched: {', '.join(cats)}"
        # Prefer the most specific match for the single-quote field
        key_quote = quotes.get(YoeCategory.multi_year.value) or quotes.get(YoeCategory.one_year.value, "(no relevant text)")

    return {
        "yoe_categories": cats,
        "yoe_reasoning": reasoning,
        "yoe_key_quote": key_quote,
        "yoe_quotes": quotes,
    }


# ── Quote verification ────────────────────────────────────────────────────

def verify_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity check that every key_quote appears in the source text."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()

    bad = []
    for _, row in df.iterrows():
        quote = row.get("yoe_key_quote", "")
        if not quote or quote == "(no relevant text)" or quote.startswith("(SES shortcut") or quote.startswith("(not a posting"):
            continue
        combined = (str(row["education"]) if pd.notna(row["education"]) else "") + " " + (
            str(row["qualification_summary"]) if pd.notna(row["qualification_summary"]) else ""
        )
        if quote in combined or norm(quote) in norm(combined):
            continue
        bad.append({
            "control_number": row["usajobs_control_number"],
            "categories": row.get("yoe_categories"),
            "quote": quote[:150],
        })
    return pd.DataFrame(bad) if bad else pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Classify a random sample of N rows")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT.name}")

    # Regex-route not_a_posting rows to their own bucket (no classification attempted)
    nap_mask = df.apply(lambda r: is_not_a_posting(r.to_dict()), axis=1)
    df_nap = df[nap_mask].copy()
    df = df[~nap_mask].reset_index(drop=True)
    print(f"  -> {len(df_nap)} not_a_posting rows routed to {NOT_A_POSTING_CATEGORY}")

    df = df[df["edu_category"].isin(ELIGIBLE_EDU)].reset_index(drop=True)
    print(f"  -> {len(df)} eligible (edu_category in {sorted(ELIGIBLE_EDU)})")

    ses_mask = df.apply(lambda r: is_ses(r.to_dict()), axis=1)
    df_ses = df[ses_mask].copy()
    df = df[~ses_mask].reset_index(drop=True)
    print(f"  -> {len(df_ses)} SES rows routed to {SES_CATEGORY}")

    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"  -> sampled {len(df)} rows for classification")

    results = [regex_classify(row.to_dict()) for _, row in df.iterrows()]

    parts = []
    if results:
        parts.append(pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1))
    if len(df_ses):
        parts.append(pd.concat(
            [df_ses.reset_index(drop=True), pd.DataFrame([SES_SHORTCUT] * len(df_ses))],
            axis=1,
        ))
    if len(df_nap):
        parts.append(pd.concat(
            [df_nap.reset_index(drop=True), pd.DataFrame([NOT_A_POSTING_SHORTCUT] * len(df_nap))],
            axis=1,
        ))

    if not parts:
        print("Nothing to write.")
        return

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(out)} YOE-classified rows to {OUTPUT}")

    cnt = Counter()
    for cats in out["yoe_categories"]:
        for c in (list(cats) if cats is not None else []):
            cnt[c] += 1
    print("\nPer-category row counts (multi-label, may sum > total):")
    for cat in [YoeCategory.no_experience.value, YoeCategory.one_year.value,
                YoeCategory.multi_year.value, SES_CATEGORY, NOT_A_POSTING_CATEGORY]:
        print(f"  {cat:25s}  {cnt.get(cat, 0)}")

    bad = verify_quotes(out)
    if len(bad) == 0:
        print("\nAll quotes verified OK")
    else:
        print(f"\n{len(bad)} bad quotes (not found in source text):")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
