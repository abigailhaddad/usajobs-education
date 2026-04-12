"""Shared async LLM batch runner with retry, caching, and progress.

All three pipeline scripts (classify, verify, extract_skills) delegate
their API calls through this module so retry logic, caching, concurrency
limits, and error handling are defined exactly once.

Settings come from config.yaml (loaded once at import time).
"""

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml
from litellm import acompletion
from pydantic import BaseModel
from tqdm import tqdm

T = TypeVar("T", bound=BaseModel)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG: dict[str, Any] = yaml.safe_load(_CONFIG_PATH.read_text())

MAX_CONCURRENT: int = CONFIG.get("max_concurrent", 5)
MAX_RETRIES: int = CONFIG.get("max_retries", 5)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """True for transient errors worth retrying (rate limits, timeouts).
    False for billing/quota failures or unrelated exceptions."""
    s = str(exc).lower()
    billing = (
        "insufficient_quota",
        "insufficient quota",
        "exceeded your current quota",
        "check your plan and billing",
    )
    if any(m in s for m in billing):
        return False
    return any(k in s for k in (
        "rate", "429", "timeout", "timed out", "connection", "overloaded",
    ))


async def call_llm(
    model: str,
    messages: list[dict],
    response_format: type[T],
) -> T:
    """Single LLM call with exponential-backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await acompletion(
                model=model,
                messages=messages,
                response_format=response_format,
            )
            return response_format.model_validate_json(
                resp.choices[0].message.content
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1 or not _is_retryable(e):
                raise
            delay = min(60, (2 ** attempt) + random.random())
            await asyncio.sleep(delay)
    raise RuntimeError("retry loop exited without returning or raising")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_key(*parts: str, version: str = "") -> str:
    combined = version + "|" + "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    items: list,
    process_fn: Callable,       # async (item, Semaphore) -> dict
    cache: dict,
    cache_key_fn: Callable,     # (item) -> str
    cache_path: Path,
    desc: str = "Processing",
    save_interval: int = 100,
) -> tuple[list[dict | None], list[int]]:
    """Run process_fn over items in parallel with caching and progress.

    Returns (results, failed_indices).  Failed items are NOT cached and
    their slot stays None.  Caller should drop them from output so a
    re-run retries them cleanly.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[dict | None] = [None] * len(items)
    failed_indices: list[int] = []
    cached_count = 0
    uncached_indices: list[int] = []

    for i, item in enumerate(items):
        key = cache_key_fn(item)
        if key in cache:
            results[i] = cache[key]
            cached_count += 1
        else:
            uncached_indices.append(i)

    print(f"  {cached_count} cached, {len(uncached_indices)} need API calls")

    if not uncached_indices:
        return results, failed_indices

    pbar = tqdm(total=len(uncached_indices), desc=desc, unit="item")
    api_since_save = 0

    async def _process_one(idx: int) -> None:
        nonlocal api_since_save
        try:
            result = await process_fn(items[idx], semaphore)
        except Exception as e:
            tqdm.write(f"  ERROR: {e}")
            failed_indices.append(idx)
            api_since_save += 1
            pbar.update(1)
            return

        key = cache_key_fn(items[idx])
        cache[key] = result
        results[idx] = result
        api_since_save += 1
        pbar.update(1)

        if api_since_save >= save_interval:
            save_cache(cache, cache_path)
            api_since_save = 0
            tqdm.write(f"  Cache saved")

    batch_size = MAX_CONCURRENT * 2
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch = uncached_indices[batch_start: batch_start + batch_size]
        await asyncio.gather(*[_process_one(idx) for idx in batch])

    pbar.close()
    save_cache(cache, cache_path)
    n_ok = len(uncached_indices) - len(failed_indices)
    print(f"  Done ({cached_count} cached, {n_ok} API OK, {len(failed_indices)} failed)")

    return results, failed_indices
