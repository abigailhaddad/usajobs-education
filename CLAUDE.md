# CLAUDE.md

Project context for Claude Code sessions in this repo.

## What this is

Analysis of whether federal 2210 (IT Specialist) job postings require a
degree. Results live in `site/data.json` and are rendered by
`site/index.html` on Netlify.

## Architecture

```
config.yaml         ← models, prompts, concurrency, retries (single source of truth)
llm_batch.py        ← shared async LLM runner: retry w/ backoff, caching, batching
classify.py         ← thin wrapper: education category classification
verify.py           ← thin wrapper: second-pass verification on flagged rows
fetch_data.py       ← pulls current_jobs from R2 → data/2210_raw.parquet
fetch_historical.py ← scrapes USAJobs pages → data/2210_historical_raw.parquet
test_classify.py    ← 22 hand-labeled test cases across all 4 categories
```

## Categories (4, mutually exclusive)

- `no_education` — education is not a qualifying path at any grade
- `education_substitutable` — education is one way to qualify but not mandatory
- `education_required` — degree is mandatory, no experience-only alternative
- `not_a_posting` — DHA notice / placeholder

Key distinction: "no substitution of education at GS-12" means education
DOES NOT HELP at that grade (no_education or education_substitutable if
other grades accept education). It does NOT mean education is required.

## Two data sources, one pipeline

Current-cycle postings come from the USAJobs API (`current_jobs_*.parquet` on
R2), which directly supplies Education and QualificationSummary fields.
Older postings are backfilled by `fetch_historical.py`, which scrapes each
announcement page and regex-extracts the `<h3>Education</h3>` and
`<h3>Qualifications</h3>` sections.

`classify.py` merges both sources at its input load step and tags each row
with a `data_source` column (`"api"` vs `"scraped"`). This column rides
through the pipeline into `site/data.json`.

## Config and caches

All LLM settings (model, prompt, concurrency, retries) live in `config.yaml`.
Each prompt has a `prompt_version` string that's folded into cache keys, so
bumping the version invalidates old cached classifications without deleting
the cache file.

Cache files in `data/`:
- `classification_cache.json` — keyed on `prompt_version + education + qualification_summary`
- `verification_cache.json` — keyed on `prompt_version + education + qualification_summary + edu_category`

## GitHub Actions

`.github/workflows/fetch_historical.yml` — manual `workflow_dispatch` with
`years`, `limit`, and `delay` inputs. Loops through years sequentially in
one job (each year reads the parquet from the previous iteration). Resume
across runs via `actions/cache` with versioned key prefix. Output is a
90-day artifact named `2210_historical_raw`.

## Gotchas

- `data/` is gitignored — everything there is local-only and must be
  regenerated or downloaded from workflow artifacts.
- `site/data.json` needs to be generated from the verified parquet and
  copied into `site/` for the Netlify deployment.
- `gh` may be authenticated as `abigailhaddad-2` by default; this repo is
  under `abigailhaddad`, so `gh auth switch --user abigailhaddad` before
  pushing or triggering workflows.
- gpt-5 family models do not support `temperature=0`. The `call_llm` function
  in `llm_batch.py` omits the temperature parameter.
- Retry logic in `llm_batch.py` distinguishes transient rate limits (retry)
  from billing/quota failures (fail fast). Failed rows are NOT cached and
  are dropped from output so re-running retries them.

## Active accounts / auth

Repo: `abigailhaddad/usajobs-education`. Main branch: `main`.
R2 base (read-only, public): `https://pub-317c58882ec04f329b63842c1eb65b0c.r2.dev/data/`.
