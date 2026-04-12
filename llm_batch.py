"""Shared LLM runner with retry, caching, real-time and batch modes.

Settings come from config.yaml (loaded once at import time).

Two execution modes:
- run_batch(): real-time async calls with concurrency control
- submit_batch() / poll_batch() / collect_batch(): OpenAI Batch API
  for 50% cost savings (results within 24h)
"""

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

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


# ---------------------------------------------------------------------------
# OpenAI Batch API (50% off, async processing)
# ---------------------------------------------------------------------------

def _get_openai_client():
    import openai
    import httpx
    return openai.Client(
        timeout=httpx.Timeout(300.0, connect=60.0),  # 5min read, 1min connect
    )


MAX_BATCH_FILE_BYTES = 10_000_000  # 10MB chunks — small enough to upload on flaky connections


def submit_batch(
    items: list,
    messages_fn: Callable,      # (item) -> list[dict] (messages array)
    cache: dict,
    cache_key_fn: Callable,     # (item) -> str
    model: str,
    response_format_schema: dict,   # JSON schema for structured output
    batch_dir: Path,
) -> tuple[list[str], list[int], list[int]]:
    """Write JSONL(s) of uncached requests, upload to OpenAI Batch API.

    Automatically splits into multiple batches if the JSONL exceeds
    the 200MB file-size limit.

    Returns (batch_ids, uncached_indices, cached_indices).
    batch_ids is empty if everything was cached.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    uncached = []
    cached_indices = []

    for i, item in enumerate(items):
        key = cache_key_fn(item)
        if key in cache:
            cached_indices.append(i)
        else:
            uncached.append(i)

    print(f"  {len(cached_indices)} cached, {len(uncached)} need API calls")

    if not uncached:
        return [], uncached, cached_indices

    raw_model = model.removeprefix("openai/")

    # Build request lines and split into chunks under the size limit
    chunks: list[list[tuple[int, str]]] = [[]]  # list of (idx, json_line) lists
    current_size = 0

    for idx in uncached:
        request = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": raw_model,
                "messages": messages_fn(items[idx]),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": response_format_schema,
                    },
                },
            },
        }
        line = json.dumps(request) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if current_size + line_bytes > MAX_BATCH_FILE_BYTES and chunks[-1]:
            chunks.append([])
            current_size = 0

        chunks[-1].append((idx, line))
        current_size += line_bytes

    print(f"  Split into {len(chunks)} batch(es): {[len(c) for c in chunks]} requests")

    # Submit each chunk
    client = _get_openai_client()
    batch_ids = []
    all_meta_chunks = []

    for chunk_i, chunk in enumerate(chunks):
        jsonl_path = batch_dir / f"batch_input_{chunk_i}.jsonl"
        chunk_indices = []
        with open(jsonl_path, "w") as f:
            for idx, line in chunk:
                f.write(line)
                chunk_indices.append(idx)

        # Upload with retry (large files can hit transient SSL/connection errors)
        for upload_attempt in range(3):
            try:
                with open(jsonl_path, "rb") as f:
                    uploaded = client.files.create(file=f, purpose="batch")
                break
            except Exception as e:
                if upload_attempt == 2:
                    raise
                print(f"  Upload error (attempt {upload_attempt + 1}): {e}")
                time.sleep(5 * (upload_attempt + 1))
        print(f"  Chunk {chunk_i}: {len(chunk)} requests, file={uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"    batch={batch.id}  status={batch.status}")
        batch_ids.append(batch.id)
        all_meta_chunks.append({
            "batch_id": batch.id,
            "file_id": uploaded.id,
            "n_requests": len(chunk),
            "uncached_indices": chunk_indices,
        })

    # Save metadata for resume
    meta = {
        "batch_ids": batch_ids,
        "chunks": all_meta_chunks,
        "total_uncached": len(uncached),
        "uncached_indices": uncached,
    }
    (batch_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2))

    return batch_ids, uncached, cached_indices


def poll_batch(batch_id: str, interval: int = 30, timeout: int = 86400) -> dict:
    """Poll until batch completes. Returns the batch object as a dict."""
    client = _get_openai_client()
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        print(f"  Batch {batch_id}: {status}  ({completed}/{total} done)", end="\r")

        if status in ("completed", "failed", "cancelled", "expired"):
            print()
            return batch.model_dump()

        if time.time() - start > timeout:
            print()
            raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

        time.sleep(interval)


def collect_batch(
    batch_result: dict,
    items: list,
    uncached_indices: list[int],
    cached_indices: list[int],
    cache: dict,
    cache_key_fn: Callable,
    cache_path: Path,
    parse_fn: Callable,     # (response_content: str) -> dict
) -> tuple[list[dict | None], list[int]]:
    """Download batch results, parse, update cache.

    Returns (results, failed_indices) same shape as run_batch().
    """
    results: list[dict | None] = [None] * len(items)
    failed_indices: list[int] = []

    # Fill in cached results
    for i in cached_indices:
        key = cache_key_fn(items[i])
        results[i] = cache[key]

    if batch_result["status"] != "completed":
        print(f"  Batch status: {batch_result['status']} — not collecting results")
        failed_indices = list(uncached_indices)
        return results, failed_indices

    # Download output file
    client = _get_openai_client()
    output_file_id = batch_result["output_file_id"]
    content = client.files.content(output_file_id).text

    parsed = 0
    errors = 0
    for line in content.strip().split("\n"):
        row = json.loads(line)
        idx = int(row["custom_id"])
        resp = row.get("response", {})

        if resp.get("status_code") != 200:
            errors += 1
            failed_indices.append(idx)
            continue

        try:
            body = resp["body"]
            msg_content = body["choices"][0]["message"]["content"]
            result = parse_fn(msg_content)
            results[idx] = result
            key = cache_key_fn(items[idx])
            cache[key] = result
            parsed += 1
        except Exception as e:
            errors += 1
            failed_indices.append(idx)

    save_cache(cache, cache_path)
    print(f"  Collected: {parsed} OK, {errors} failed, {len(cached_indices)} cached")
    return results, failed_indices
