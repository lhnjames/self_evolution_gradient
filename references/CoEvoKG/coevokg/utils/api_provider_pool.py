"""Generic API provider pool with per-model/key slots and cooldowns."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from coevokg.utils.env import get_coevokg_env

logger = logging.getLogger(__name__)


class FatalProviderPoolExhausted(RuntimeError):
    """Raised when all configured providers are unavailable for too long."""


class BadRequestNoRetry(RuntimeError):
    """Raised for deterministic request errors that should not be retried."""


@dataclass
class Provider:
    model: str
    key_name: str
    api_key: str
    slots: int
    client: OpenAI
    semaphore: threading.BoundedSemaphore = field(init=False)
    inflight: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    disabled: bool = False
    quota_exhausted: bool = False

    def __post_init__(self) -> None:
        self.semaphore = threading.BoundedSemaphore(max(1, self.slots))


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_keys() -> list[tuple[str, str]]:
    raw = get_coevokg_env("API_KEYS", "").strip()
    if raw:
        return [(f"key{i + 1}", k) for i, k in enumerate(_split_csv(raw))]

    keys = []
    for suffix in ("API_KEY", "API_KEY_2"):
        key = get_coevokg_env(suffix, "").strip()
        if key:
            keys.append((f"COEVOKG_{suffix}", key))
    return keys


def _parse_model_slots() -> list[tuple[str, int]]:
    raw = get_coevokg_env(
        "MODEL_SLOT_TOTALS",
        f"{get_coevokg_env('MODEL', 'judge-model')}:16",
    )
    parsed: list[tuple[str, int]] = []
    for item in _split_csv(raw):
        if ":" not in item:
            continue
        model, slots = item.rsplit(":", 1)
        try:
            n_slots = int(slots)
        except ValueError:
            continue
        if model.strip() and n_slots > 0:
            parsed.append((model.strip(), n_slots))
    return parsed


def _status_and_code(exc: Exception) -> tuple[Optional[int], str]:
    status = getattr(exc, "status_code", None)
    code = ""
    if isinstance(exc, APIStatusError):
        try:
            data = exc.response.json()
            err = data.get("error", {}) if isinstance(data, dict) else {}
            code = str(err.get("code") or err.get("type") or "")
        except Exception:
            code = ""
    return status, code


class APIProviderPool:
    """A shared pool that distributes calls across model/key providers.

    The COEVOKG_MODEL_SLOT_TOTALS environment variable controls per-model slots, for example "model-a:8,model-b:8".
    """

    def __init__(self, name: str, base_url: str, timeout: float):
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._providers = self._build_providers()
        self._no_provider_since: Optional[float] = None
        self._consecutive_no_provider = 0
        self._fatal_after_s = float(get_coevokg_env("POOL_FATAL_AFTER_S", "600"))
        self._fatal_after_count = int(get_coevokg_env("POOL_FATAL_AFTER_COUNT", "200"))
        self._max_attempts = int(get_coevokg_env("PROVIDER_MAX_ATTEMPTS", "3"))

        if not self._providers:
            raise ValueError("No API providers configured. Set COEVOKG_API_KEY/COEVOKG_API_KEY_2 or COEVOKG_API_KEYS.")
        logger.info(
            "[%s] APIProviderPool initialized: %s",
            self.name,
            ", ".join(f"{p.model}/{p.key_name}:{p.slots}" for p in self._providers),
        )

    def _build_providers(self) -> list[Provider]:
        keys = _load_keys()
        models = _parse_model_slots()
        providers: list[Provider] = []
        if not keys or not models:
            return providers

        for model, total_slots in models:
            base = total_slots // len(keys)
            extra = total_slots % len(keys)
            for i, (key_name, api_key) in enumerate(keys):
                slots = base + (1 if i < extra else 0)
                if slots <= 0:
                    continue
                client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout, max_retries=0)
                providers.append(Provider(model=model, key_name=key_name, api_key=api_key, slots=slots, client=client))
        return providers

    def _healthy_candidates(self, exclude: set[tuple[str, str]]) -> list[Provider]:
        now = time.time()
        return [
            p for p in self._providers
            if (p.model, p.key_name) not in exclude
            and not p.disabled
            and now >= p.cooldown_until
            and p.semaphore.acquire(blocking=False)
        ]

    def _release_candidate(self, provider: Provider) -> None:
        try:
            provider.semaphore.release()
        except ValueError:
            pass

    def _choose_provider(self, exclude: set[tuple[str, str]]) -> Provider:
        acquire_wait = float(get_coevokg_env("POOL_ACQUIRE_WAIT", "30"))
        deadline = time.time() + acquire_wait
        while True:
            candidates = self._healthy_candidates(exclude)
            if candidates:
                with self._lock:
                    self._no_provider_since = None
                    self._consecutive_no_provider = 0
                # Weighted by configured slots; semaphore caps enforce the hard per-provider concurrency.
                provider = random.choices(candidates, weights=[p.slots for p in candidates], k=1)[0]
                for p in candidates:
                    if p is not provider:
                        self._release_candidate(p)
                with self._lock:
                    provider.inflight += 1
                return provider

            self._record_no_provider()
            if time.time() >= deadline:
                raise RuntimeError(f"[{self.name}] no available API provider for {acquire_wait:.0f}s")
            time.sleep(0.05)

    def _record_no_provider(self) -> None:
        now = time.time()
        with self._lock:
            if self._no_provider_since is None:
                self._no_provider_since = now
            self._consecutive_no_provider += 1
            elapsed = now - self._no_provider_since
            all_quota_or_disabled = all(p.disabled or p.quota_exhausted for p in self._providers)
            if all_quota_or_disabled and (
                elapsed >= self._fatal_after_s or self._consecutive_no_provider >= self._fatal_after_count
            ):
                raise FatalProviderPoolExhausted(
                    f"[{self.name}] all API providers exhausted/disabled "
                    f"(elapsed={elapsed:.1f}s, count={self._consecutive_no_provider})"
                )

    def _mark_success(self, provider: Provider) -> None:
        with self._lock:
            provider.consecutive_failures = 0
            provider.quota_exhausted = False
            provider.inflight = max(0, provider.inflight - 1)
        self._release_candidate(provider)

    def _mark_failure(self, provider: Provider, exc: Exception) -> None:
        status, code = _status_and_code(exc)
        message = str(exc).lower()
        now = time.time()
        cooldown = 10.0
        fatal_for_request = False

        with self._lock:
            provider.consecutive_failures += 1
            provider.inflight = max(0, provider.inflight - 1)

            if status == 400:
                fatal_for_request = True
            elif status == 401:
                provider.disabled = True
            elif status == 403:
                if "insufficient_quota" in code:
                    provider.quota_exhausted = True
                    cooldown = float(get_coevokg_env("PROVIDER_COOLDOWN_QUOTA_S", str(6 * 3600)))
                else:
                    provider.disabled = True
            elif status == 429:
                cooldown = float(get_coevokg_env("PROVIDER_COOLDOWN_429_S", "30"))
            elif (
                status in (500, 502, 503)
                or isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError))
                or "empty chat completion" in message
            ):
                cooldown = float(get_coevokg_env("PROVIDER_COOLDOWN_TRANSIENT_S", "10"))

            if not provider.disabled and not fatal_for_request:
                provider.cooldown_until = max(provider.cooldown_until, now + cooldown)

        self._release_candidate(provider)

        logger.warning(
            "[%s] provider failed model=%s key=%s status=%s code=%s disabled=%s quota=%s cooldown=%.0fs error=%s",
            self.name,
            provider.model,
            provider.key_name,
            status,
            code,
            provider.disabled,
            provider.quota_exhausted,
            max(0.0, provider.cooldown_until - now),
            exc,
        )

        if fatal_for_request:
            raise BadRequestNoRetry(f"Bad request from {provider.model}/{provider.key_name}: {exc}") from exc

    def call(self, fn: Callable[[OpenAI, str], Any]) -> Any:
        tried: set[tuple[str, str]] = set()
        last_exc: Optional[Exception] = None

        for _ in range(max(1, self._max_attempts)):
            provider = self._choose_provider(exclude=tried)
            tried.add((provider.model, provider.key_name))
            try:
                result = fn(provider.client, provider.model)
                self._mark_success(provider)
                return result
            except BadRequestNoRetry:
                self._mark_success(provider)
                raise
            except Exception as exc:
                last_exc = exc
                self._mark_failure(provider, exc)

        raise RuntimeError(f"[{self.name}] API provider call failed after {self._max_attempts} provider attempts") from last_exc


_POOLS: dict[tuple[str, str, float], APIProviderPool] = {}
_POOLS_LOCK = threading.Lock()


def get_api_provider_pool(name: str, base_url: str, timeout: float) -> APIProviderPool:
    key = (name, base_url, float(timeout))
    with _POOLS_LOCK:
        if key not in _POOLS:
            _POOLS[key] = APIProviderPool(name=name, base_url=base_url, timeout=timeout)
        return _POOLS[key]
