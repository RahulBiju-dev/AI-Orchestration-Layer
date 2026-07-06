"""Return the current local or timezone-specific date and time."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_current_datetime(timezone: str | None = None) -> str:
    """Return a stable, machine-readable snapshot of the current date and time."""
    requested_timezone = str(timezone or "").strip()
    try:
        if requested_timezone:
            zone = ZoneInfo(requested_timezone)
            now = datetime.now(zone)
            timezone_name = requested_timezone
        else:
            now = datetime.now().astimezone()
            timezone_name = getattr(now.tzinfo, "key", None) or now.tzname() or "local"
    except (ZoneInfoNotFoundError, ValueError):
        return json.dumps({
            "error": f"Unknown IANA timezone: {requested_timezone}",
            "example": "Asia/Kolkata",
        })

    offset = now.strftime("%z")
    formatted_offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return json.dumps({
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "timezone": timezone_name,
        "utc_offset": formatted_offset,
        "unix_timestamp": int(now.timestamp()),
    }, ensure_ascii=False)
