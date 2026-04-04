# USAJobs 2210 Education Requirements

Do federal IT jobs actually require a degree? This project classifies education requirements across ~3,900 2210 (IT Specialist) job postings from USAJobs.

**TL;DR: 99.2% do not require a degree.**

## Pipeline

1. **`fetch_data.py`** — Pulls current_jobs parquet files from R2 and extracts 2210 series jobs with education/qualification fields
2. **`classify.py`** — First-pass classification using GPT-5.4-mini with structured output
3. **`verify.py`** — Second-pass verification using GPT-5.4 on all non-obvious classifications and suspicious patterns
4. **`site/`** — Static site with results, deployed on Netlify

## Categories

- **no_education** — No degree required at any grade level (includes "education can substitute for experience" since experience alone qualifies)
- **education_required** — A bachelor's degree or higher is explicitly required with no experience-only alternative
- **education_required_higher** — Degree required only at higher grade levels
- **not_a_posting** — DHA notices, resume collection, not actual job postings

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python fetch_data.py
python classify.py
python verify.py
```

## Data

Source: [USAJobs](https://www.usajobs.gov) current_jobs parquet files (2024-2026) stored in Cloudflare R2.
