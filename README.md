# USAJobs 2210 Education Requirements

Do federal IT jobs actually require a degree? This project classifies education requirements across 2210 (IT Specialist) job postings from USAJobs.

## Pipeline

1. **`fetch_data.py`** — Pulls `current_jobs` parquet files from R2 and extracts 2210 series jobs with education/qualification fields → `data/2210_raw.parquet`
2. **`fetch_historical.py`** — Backfills older 2210 postings that rolled out of `current_jobs`. The `historical_jobs` parquets on R2 carry metadata only, so this scrapes each announcement page and extracts Education + Qualifications sections via regex → `data/2210_historical_raw.parquet`. Runs via the `Fetch historical 2210 jobs` GitHub Actions workflow (loops through years sequentially, resumes via `actions/cache`).
3. **`classify.py`** — First-pass classification using structured output. Unions `2210_raw.parquet` + `2210_historical_raw.parquet` (when present) and tags each row with `data_source` (`"api"` / `"scraped"`) → `data/2210_classified.parquet`
4. **`verify.py`** — Second-pass verification using a stronger model on flagged rows → `data/2210_verified.parquet`
5. **`site/`** — Static site with results, deployed on Netlify

Models, prompts, concurrency, and retry settings live in **`config.yaml`**. Shared LLM runner (retry with backoff, caching, async batching) lives in **`llm_batch.py`**.

## Categories

- **no_education** — Education is not a qualifying path at any grade level. You must have experience; a degree alone will not get you in.
- **education_substitutable** — Education is not mandatory, but the posting offers it as one way to qualify alongside experience. A candidate with the right degree and no experience can qualify. This is the shape of most 2210 postings under OPM Alternative A.
- **education_required** — A degree is explicitly required with no experience-only alternative.
- **not_a_posting** — DHA notices, resume collection, not actual job postings.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python fetch_data.py
python classify.py
python verify.py
```

### Backfilling historical postings

Trigger the `Fetch historical 2210 jobs` workflow on GitHub Actions (manual
`workflow_dispatch`). It loops through each year sequentially and uploads
the cumulative parquet as an artifact. When it completes:

```bash
gh run download <run_id> --repo abigailhaddad/usajobs-education \
    --name 2210_historical_raw --dir data/
python classify.py
python verify.py
```

### Tests

```bash
python test_classify.py   # 22 hand-labeled cases across all 4 categories
```

## Data

Source: [USAJobs](https://www.usajobs.gov) `current_jobs` and `historical_jobs`
parquet files (2023–2026) stored in Cloudflare R2. Historical postings are
backfilled by scraping the USAJobs announcement page and regex-extracting the
Education + Qualifications `<h3>` sections.
