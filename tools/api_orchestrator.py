"""Resilient HTTP API execution with bounded retries and secret-safe auth."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


def _secret(name: str | None) -> str | None:
    return os.environ.get(name, "") if name else None


def _prepare_auth(auth: dict, headers: dict[str, str], timeout: float) -> tuple[Any, dict[str, str]]:
    kind = auth.get("type", "none")
    request_auth = None
    if kind == "bearer":
        token = _secret(auth.get("token_env"))
        if not token:
            raise ValueError("Bearer token environment variable is unset")
        headers["Authorization"] = f"Bearer {token}"
    elif kind == "api_key":
        value = _secret(auth.get("value_env"))
        if not value:
            raise ValueError("API key environment variable is unset")
        headers[str(auth.get("header", "X-API-Key"))] = value
    elif kind == "basic":
        username = _secret(auth.get("username_env"))
        password = _secret(auth.get("password_env"))
        if not username or not password:
            raise ValueError("Basic auth environment variables are unset")
        request_auth = (username, password)
    elif kind == "oauth2_client_credentials":
        token_url = str(auth.get("token_url", ""))
        client_id = _secret(auth.get("client_id_env"))
        client_secret = _secret(auth.get("client_secret_env"))
        if not token_url or not client_id or not client_secret:
            raise ValueError("OAuth token_url/client environment variables are required")
        response = requests.post(
            token_url,
            data={"grant_type": "client_credentials", "scope": auth.get("scope", "")},
            auth=(client_id, client_secret), timeout=timeout,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("OAuth response did not contain access_token")
        headers["Authorization"] = f"Bearer {token}"
    elif kind != "none":
        raise ValueError(f"Unsupported auth type: {kind}")
    return request_auth, headers


def _documentation_suggestions(primary: str, documentation: dict | None) -> list[str]:
    urls: list[str] = []
    if documentation and isinstance(documentation.get("paths"), dict):
        base = str(documentation.get("base_url") or f"{urlparse(primary).scheme}://{urlparse(primary).netloc}")
        for path, definition in documentation["paths"].items():
            deprecated = isinstance(definition, dict) and definition.get("deprecated") is True
            candidate = urljoin(base.rstrip("/") + "/", str(path).lstrip("/"))
            if not deprecated and candidate not in urls:
                urls.append(candidate)
    return urls[:10]


def api_orchestrator(
    request: dict,
    auth: dict | None = None,
    retry: dict | None = None,
    alternative_endpoints: list[str] | None = None,
    documentation: dict | None = None,
) -> str:
    """Execute an API call with auth refresh, backoff, and endpoint failover."""
    method = str(request.get("method", "GET")).upper()
    primary = str(request.get("url", ""))
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return json.dumps({"error": f"Unsupported HTTP method: {method}"})
    if urlparse(primary).scheme not in {"http", "https"}:
        return json.dumps({"error": "request.url must use http or https"})
    policy = retry or {}
    attempts = max(1, min(int(policy.get("max_attempts", 3)), 6))
    base_delay = max(0.0, min(float(policy.get("base_delay", 0.5)), 10.0))
    timeout = max(0.5, min(float(request.get("timeout", 20)), 120.0))
    max_chars = max(1000, min(int(request.get("max_response_chars", 20000)), 100000))
    urls = [primary] + [str(url) for url in (alternative_endpoints or [])]
    urls = urls[:10]
    invalid_urls = [url for url in urls if urlparse(url).scheme not in {"http", "https"}]
    if invalid_urls:
        return json.dumps({"error": "All alternative endpoints must use http or https", "invalid": invalid_urls})
    suggestions = _documentation_suggestions(primary, documentation)
    audit: list[dict] = []

    headers = {str(key): str(value) for key, value in request.get("headers", {}).items()}
    try:
        request_auth, headers = _prepare_auth(auth or {}, headers, timeout)
    except Exception as exc:
        return json.dumps({"error": str(exc), "secret_policy": "Credentials must be supplied through environment-variable names"})

    last_error = "No request attempted"
    for endpoint_index, endpoint in enumerate(urls):
        for attempt in range(1, attempts + 1):
            try:
                response = requests.request(
                    method, endpoint, headers=headers, params=request.get("params"),
                    json=request.get("json"), data=request.get("data"), auth=request_auth,
                    timeout=timeout, allow_redirects=bool(request.get("allow_redirects", True)),
                )
                deprecation_header = response.headers.get("Deprecation", "").strip().lower()
                deprecated = response.status_code == 410 or bool(response.headers.get("Sunset")) or deprecation_header not in {"", "false", "0"}
                audit.append({"endpoint": endpoint, "attempt": attempt, "status": response.status_code, "deprecated": deprecated})
                if response.status_code == 401 and (auth or {}).get("type") == "oauth2_client_credentials" and attempt < attempts:
                    request_auth, headers = _prepare_auth(auth or {}, headers, timeout)
                    audit[-1]["auth_refreshed"] = True
                    continue
                if response.status_code not in RETRYABLE and response.status_code != 404 and not deprecated:
                    body = response.text
                    return json.dumps({
                        "ok": response.ok, "status": response.status_code, "endpoint": endpoint,
                        "headers": {key: value for key, value in response.headers.items() if key.lower() not in {"set-cookie", "authorization"}},
                        "body": body[:max_chars], "truncated": len(body) > max_chars, "attempts": audit,
                    }, ensure_ascii=False)
                last_error = f"HTTP {response.status_code}"
                if deprecated or response.status_code in {404, 410}:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 30.0) if retry_after and retry_after.isdigit() else min(base_delay * (2 ** (attempt - 1)), 30.0)
                if attempt < attempts:
                    time.sleep(delay)
            except requests.RequestException as exc:
                last_error = str(exc)
                audit.append({"endpoint": endpoint, "attempt": attempt, "error": type(exc).__name__})
                if attempt < attempts:
                    time.sleep(min(base_delay * (2 ** (attempt - 1)), 30.0))
        if endpoint_index + 1 >= len(urls):
            break
    return json.dumps({"ok": False, "error": last_error, "attempts": audit, "alternatives_considered": urls[1:], "documentation_suggestions": suggestions}, ensure_ascii=False)
