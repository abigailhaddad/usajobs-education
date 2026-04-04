#!/usr/bin/env python3
"""Fetch 2210 IT jobs from R2 current_jobs parquets and extract education fields."""

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "2210_raw.parquet"


def load_parquet(year: int) -> pd.DataFrame:
    url = f"{R2_BASE}/data/current_jobs_{year}.parquet"
    print(f"Fetching {year}...")
    req = urllib.request.Request(url, headers={"User-Agent": "usajobs-download/1.0"})
    resp = urllib.request.urlopen(req)
    return pd.read_parquet(io.BytesIO(resp.read()))


def extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["MatchedObjectDescriptor"])
        details = d.get("UserArea", {}).get("Details", {})
        control = row.get("usajobsControlNumber") or row.get("usajobs_control_number", "")
        records.append({
            "usajobs_control_number": str(control),
            "usajobs_url": f"https://www.usajobs.gov/job/{control}" if control else "",
            "position_title": row.get("positionTitle", ""),
            "agency": row.get("hiringAgencyName", ""),
            "department": row.get("hiringDepartmentName", ""),
            "min_grade": row.get("minimumGrade", ""),
            "max_grade": row.get("maximumGrade", ""),
            "pay_scale": row.get("payScale", ""),
            "min_salary": row.get("minimumSalary", ""),
            "max_salary": row.get("maximumSalary", ""),
            "open_date": row.get("positionOpenDate", ""),
            "close_date": row.get("positionCloseDate", ""),
            "service_type": row.get("serviceType", ""),
            "appointment_type": row.get("appointmentType", ""),
            "education": details.get("Education", ""),
            "qualification_summary": d.get("QualificationSummary", ""),
            "source_year": row["source_year"],
        })
    return pd.DataFrame(records)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    frames = []
    for year in [2024, 2025, 2026]:
        df = load_parquet(year)
        mask = df["JobCategories"].str.contains("2210", na=False)
        subset = df[mask].copy()
        subset["source_year"] = year
        frames.append(subset)
        print(f"  {len(subset)} 2210 jobs")

    all_2210 = pd.concat(frames, ignore_index=True)
    result = extract_fields(all_2210)
    result.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {len(result)} jobs to {OUTPUT}")


if __name__ == "__main__":
    main()
