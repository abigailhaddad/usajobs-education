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
]


@pytest.mark.parametrize("text,expected", CASES)
def test_cases(text, expected):
    got = regex_classify({"education": "", "qualification_summary": text})["yoe_categories"]
    assert sorted(got) == sorted(expected)
