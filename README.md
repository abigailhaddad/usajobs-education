# USAJobs 2210 Education Requirements

Do federal IT jobs actually require a degree? This project classifies education requirements across ~3,900 2210 (IT Specialist) job postings from USAJobs.

**TL;DR: 99.2% do not require a degree.**

## Pipeline

1. **`fetch_data.py`** — Pulls `current_jobs` parquet files from R2 and extracts 2210 series jobs with education/qualification fields → `data/2210_raw.parquet`
2. **`fetch_historical.py`** — Backfills older 2210 postings that rolled out of `current_jobs` before we started pulling. The `historical_jobs` parquets on R2 carry metadata only, so this script scrapes each `https://www.usajobs.gov/job/{control_number}` announcement page and extracts the Education and Qualifications `<h3>` sections via regex → `data/2210_historical_raw.parquet`. Runs via the `Fetch historical 2210 jobs` GitHub Actions workflow; the output parquet is published as a run artifact.
3. **`classify.py`** — First-pass classification using GPT-5.4-mini with structured output. Unions `2210_raw.parquet` + `2210_historical_raw.parquet` (when present) and tags each row with a `data_source` column (`"api"` vs `"scraped"`) → `data/2210_classified.parquet`
4. **`verify.py`** — Second-pass verification using GPT-5.4 on all non-`no_education` classifications and `no_education` rows that match suspicious regex patterns → `data/2210_verified.parquet`
5. **`extract_skills.py`** — Extracts structured skills/specialization/certs per posting using GPT-5.4-mini → `data/2210_skills.json` (copied to `site/data.json`)
6. **`site/`** — Static site with results, deployed on Netlify

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
python extract_skills.py
cp data/2210_skills.json site/data.json
```

### Backfilling historical postings

Trigger the `Fetch historical 2210 jobs` workflow on GitHub Actions (manual
`workflow_dispatch`, optional `limit` input for dry runs). When it completes,
pull the artifact into `data/` and re-run the pipeline:

```bash
gh run download <run_id> --repo abigailhaddad/usajobs-education \
    --name 2210_historical_raw --dir data/
python classify.py        # cache-hits existing API rows, only scrapes new
python verify.py
python extract_skills.py
cp data/2210_skills.json site/data.json
```

## Data

Source: [USAJobs](https://www.usajobs.gov) `current_jobs` and `historical_jobs`
parquet files (2024–2026) stored in Cloudflare R2. Historical postings that
are no longer in `current_jobs` are backfilled by scraping the USAJobs
announcement page and regex-extracting the Education + Qualifications
sections — see `fetch_historical.py` and `validate_scrape.py` for the
validation harness that checks scraped extractions against API ground truth.
