"""
tools/web_scraper.py — bounded web page extraction for agent research.

The scraper is intentionally read-only and conservative: it fetches public
HTTP(S) pages, rejects localhost/private-network targets, follows a small
number of redirects, and returns compact extracted text plus page metadata.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse

MAX_BYTES = 2_000_000
MAX_CHARS = 20_000
MAX_LINKS = 40
MAX_HEADINGS = 40
MAX_REDIRECTS = 5
TIMEOUT = (5, 15)

USER_AGENT = (
    "SeleneAgent/1.0 (+https://localhost; bounded research scraper) "
    "Mozilla/5.0"
)

_WHITESPACE_RE = re.compile(r"\s+")
_TEXTUAL_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "text/xml",
    "application/xhtml+xml",
    "application/xml",
    "application/json",
    "application/ld+json",
)


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _clean_text(value: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", html.unescape(value or "")).strip()
    cleaned = re.sub(r"\s+([.,!?;:%)\]\}])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[\{])\s+", r"\1", cleaned)
    return cleaned


def _looks_textual(content_type: str) -> bool:
    lower = content_type.lower().split(";", 1)[0].strip()
    return lower.startswith("text/") or lower in _TEXTUAL_TYPES


def _validate_public_http_url(url: str) -> str:
    """Return a normalized URL or raise ValueError when unsafe/unsupported."""
    url = str(url or "").strip()
    if not url:
        raise ValueError("url is required")

    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("only http and https URLs can be scraped")
    if not parsed.hostname:
        raise ValueError("URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed")

    host = parsed.hostname.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise ValueError("localhost targets are not allowed")

    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    addresses: set[ipaddress._BaseAddress] = set()
    if literal_ip is not None:
        addresses.add(literal_ip)
    else:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"could not resolve host: {exc}") from exc
        for info in infos:
            address = info[4][0]
            try:
                addresses.add(ipaddress.ip_address(address))
            except ValueError:
                continue

    if not addresses:
        raise ValueError("could not resolve host to an IP address")

    for address in addresses:
        if not address.is_global:
            raise ValueError("private, local, reserved, or otherwise non-public targets are not allowed")

    safe_url, _fragment = urldefrag(url)
    return safe_url


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.description = ""
        self.canonical_url = ""
        self.lang = ""
        self.text_parts: list[str] = []
        self.headings: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self._skip_stack: list[str] = []
        self._capture_title = False
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): (value or "") for key, value in attrs}

        if tag in {"script", "style", "noscript", "template", "svg", "canvas"}:
            self._skip_stack.append(tag)
            return

        if tag == "html" and attrs_dict.get("lang"):
            self.lang = _clean_text(attrs_dict["lang"])
        elif tag == "title":
            self._capture_title = True
        elif tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            content = attrs_dict.get("content", "")
            if content and (name == "description" or prop == "og:description"):
                self.description = self.description or _clean_text(content)
            elif content and prop == "og:title":
                self.title = self.title or _clean_text(content)
        elif tag == "link" and attrs_dict.get("rel", "").lower() == "canonical" and attrs_dict.get("href"):
            self.canonical_url = urljoin(self.base_url, attrs_dict["href"])
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag == "a" and attrs_dict.get("href") and len(self.links) < MAX_LINKS:
            href = urljoin(self.base_url, attrs_dict["href"])
            if href.startswith(("http://", "https://")):
                self._link_href = urldefrag(href)[0]
                self._link_parts = []

        if tag in {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            return
        if self._skip_stack:
            return

        if tag == "title":
            self._capture_title = False
        elif tag == self._heading_tag:
            text = _clean_text(" ".join(self._heading_parts))
            if text and len(self.headings) < MAX_HEADINGS:
                self.headings.append({"level": self._heading_tag.upper(), "text": text})
            self._heading_tag = None
            self._heading_parts = []
        elif tag == "a" and self._link_href:
            text = _clean_text(" ".join(self._link_parts))
            if text and not any(link["url"] == self._link_href for link in self.links):
                self.links.append({"text": text[:160], "url": self._link_href})
            self._link_href = None
            self._link_parts = []

        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        text = _clean_text(data)
        if not text:
            return
        if self._capture_title:
            self.title = _clean_text(f"{self.title} {text}") if self.title else text
        if self._heading_tag:
            self._heading_parts.append(text)
        if self._link_href:
            self._link_parts.append(text)
        self.text_parts.append(text)

    def extracted_text(self) -> str:
        paragraphs: list[str] = []
        for line in "".join(part if part == "\n" else f" {part} " for part in self.text_parts).splitlines():
            cleaned = _clean_text(line)
            if cleaned:
                paragraphs.append(cleaned)
        return "\n".join(paragraphs)


def _decode_response(content: bytes, encoding: str | None) -> str:
    if encoding:
        try:
            return content.decode(encoding, errors="replace")
        except LookupError:
            pass
    return content.decode("utf-8", errors="replace")


def _fetch(url: str) -> dict:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - dependency guidance
        return {"error": f"requests is required for web_scrape: {exc}"}

    current_url = url
    redirects: list[str] = []
    session = requests.Session()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.3",
        "Accept-Language": "en-US,en;q=0.8",
    }

    try:
        for _ in range(MAX_REDIRECTS + 1):
            safe_url = _validate_public_http_url(current_url)
            response = session.get(safe_url, headers=headers, timeout=TIMEOUT, stream=True, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location", "")
                if not location:
                    return {"error": "redirect response did not include a Location header", "url": safe_url}
                current_url = urljoin(safe_url, location)
                redirects.append(current_url)
                continue
            break
        else:
            return {"error": f"too many redirects; limit is {MAX_REDIRECTS}", "redirects": redirects}

        final_url = _validate_public_http_url(response.url)
        content_type = response.headers.get("Content-Type", "")
        if content_type and not _looks_textual(content_type):
            return {
                "error": "URL did not return a textual page",
                "url": final_url,
                "status_code": response.status_code,
                "content_type": content_type,
                "guidance": "Use a document, image, or browser-specific tool for non-text web assets.",
            }

        total = 0
        chunks: list[bytes] = []
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_BYTES:
                remaining = MAX_BYTES - sum(len(part) for part in chunks)
                if remaining > 0:
                    chunks.append(chunk[:remaining])
                break
            chunks.append(chunk)

        content = b"".join(chunks)
        text = _decode_response(content, response.encoding)
        return {
            "url": final_url,
            "requested_url": url,
            "status_code": response.status_code,
            "content_type": content_type,
            "encoding": response.encoding,
            "text": text,
            "truncated_bytes": total > MAX_BYTES,
            "redirects": redirects,
        }
    except Exception as exc:
        return {"error": str(exc), "url": current_url}
    finally:
        session.close()


def _extract(url: str, raw: dict, max_chars: int, include_links: bool) -> dict:
    content_type = str(raw.get("content_type") or "")
    text = str(raw.get("text") or "")
    is_html = "html" in content_type.lower() or re.search(r"<\s*(html|head|body|title|p|article)\b", text[:5000], re.I)

    if is_html:
        parser = _ReadableHTMLParser(str(raw.get("url") or url))
        parser.feed(text)
        parser.close()
        extracted = parser.extracted_text()
        title = parser.title
        description = parser.description
        headings = parser.headings
        links = parser.links if include_links else []
        canonical_url = parser.canonical_url or str(raw.get("url") or url)
        lang = parser.lang
    else:
        extracted = _clean_text(text)
        title = ""
        description = ""
        headings = []
        links = []
        canonical_url = str(raw.get("url") or url)
        lang = ""

    truncated_chars = len(extracted) > max_chars
    extracted = extracted[:max_chars].rstrip()

    return {
        "url": str(raw.get("url") or url),
        "canonical_url": canonical_url,
        "requested_url": str(raw.get("requested_url") or url),
        "status_code": raw.get("status_code"),
        "content_type": raw.get("content_type"),
        "title": title,
        "description": description,
        "language": lang,
        "headings": headings,
        "text": extracted,
        "links": links,
        "truncated": bool(raw.get("truncated_bytes") or truncated_chars),
        "limits": {"max_bytes": MAX_BYTES, "max_chars": max_chars},
    }


def web_scrape(url: str, max_chars: int = MAX_CHARS, include_links: bool = False) -> str:
    """
    Fetch and extract readable text from a public HTTP(S) URL.

    Args:
        url: The page URL to fetch. Bare domains are treated as HTTPS URLs.
        max_chars: Maximum extracted text characters to return (1,000-50,000).
        include_links: Include up to 40 page links with anchor text.

    Returns:
        Compact JSON containing metadata, headings, extracted text, and limits.
    """
    try:
        safe_url = _validate_public_http_url(url)
    except ValueError as exc:
        return _json({"error": str(exc)})

    max_chars = _clamp_int(max_chars, MAX_CHARS, 1_000, 50_000)
    raw = _fetch(safe_url)
    if raw.get("error"):
        return _json(raw)
    return _json(_extract(safe_url, raw, max_chars, bool(include_links)))
