"""Text-only regression cases for yoe_classify.regex_classify (no data files needed)."""
import pytest

from yoe_classify import regex_classify

CASES = [
    ("must have at least 1-year specialized experience equivalent to at least the next lower grade, GS-07.", ["yoe_one_year"]),
    ("must have one-year specialized experience equivalent to the next lower grade", ["yoe_one_year"]),
    ("must have two years specialized experience, including one year at the CL 27", ["yoe_multi_year"]),
    ("3 or more years' experience building applications.", ["yoe_multi_year"]),
    ("10 years' experience in the Information Security field", ["yoe_multi_year"]),
    ("Minimum of four years user support experience", ["yoe_multi_year"]),
    ("one year of specialized experience at GS-12", ["yoe_one_year"]),
    ("Applicants must be at least 18 years old", ["yoe_no_experience"]),
    ("a 4-year degree in computer science", ["yoe_no_experience"]),
    # Preferred / desired / not-required experience does not set the requirement.
    ("Knowledge of and three to five years' experience in computer forensic cyber analysis, or the equivalent is highly desired", ["yoe_no_experience"]),
    ("Two or more years project management experience in IT is preferred", ["yoe_no_experience"]),
    ("NOTE: 2-5 years' experience managing LAN campus networks is preferred but not required", ["yoe_no_experience"]),
    ("2 years proven experience as network technician or relevant position desired", ["yoe_no_experience"]),
    ("Preferred Qualifications: 2 - 3 years of experience providing direct customer service", ["yoe_no_experience"]),
    ("Preferred candidates would have at least 2 years of quality experience teaching advanced topics", ["yoe_no_experience"]),
    ("5 years of experience is a plus", ["yoe_no_experience"]),
    ("3 years of Python experience is nice to have", ["yoe_no_experience"]),
    ("One year of specialized experience is optional for this role. Three years of experience helpful.", ["yoe_no_experience"]),
    # Required one-year span stays when a different sentence/clause is preferred-only.
    ("You must have one year of specialized experience equivalent to the GS-11. Five years' experience is preferred.", ["yoe_one_year"]),
    ("You must have one year of specialized experience equivalent to the GS-11; three years of cloud experience desired", ["yoe_one_year"]),
    ("One year of specialized experience equivalent to GS-11.\n- 2 years of experience in cybersecurity desired", ["yoe_one_year"]),
    ("At least 1 year of specialized experience. At least three years of leadership experience desired. Minimum of 4 years user support experience.", ["yoe_multi_year", "yoe_one_year"]),
    # Same clause says both required and preferred: kept (conservative).
    ("A minimum of 3 years of experience is required, 5 years preferred", ["yoe_multi_year"]),
    ("Bachelor's degree preferred, but may be substituted by a minimum of 3 years general experience", ["yoe_multi_year"]),
    # "preferably in X" and "(... preferred)" qualify the experience; they do not make the years optional.
    ("three years of relevant computer support experience preferably in a call center", ["yoe_multi_year"]),
    ("at least five years overseeing enterprise-class programs (federal, financial, or healthcare sectors preferred)", ["yoe_multi_year"]),
    ("One year of specialized experience at the GS-12 level, preferably in support of identity management", ["yoe_one_year"]),
    # "or equivalent" wording does not make years optional.
    ("three years of experience in computer forensics, or the equivalent", ["yoe_multi_year"]),
    # Veteran's preference is not experience preference.
    ("Veterans' preference applies. One year of specialized experience equivalent to GS-9.", ["yoe_one_year"]),
    # Under a Minimum Qualifications heading, with a preferred sentence elsewhere.
    ("Minimum Qualifications: 3 years of IT experience. Preferred Qualifications: CISSP certification.", ["yoe_multi_year"]),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_cases(text, expected):
    got = regex_classify({"education": "", "qualification_summary": text})["yoe_categories"]
    assert sorted(got) == sorted(expected)
