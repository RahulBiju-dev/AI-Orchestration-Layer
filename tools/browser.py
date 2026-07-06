"""
tools/browser.py - Browser control tool.

This module provides a unified tool for opening URLs and performing web searches
in the user's default web browser.
"""
import urllib.parse
import webbrowser
import re


# Web apps do not have an OS-level metadata registry like Linux .desktop files.
# Keep this list limited to branded, unambiguous names so normal search queries
# do not unexpectedly turn into site launches.
_WEB_APPS = {
    "https://calendar.google.com/": ("google calendar",),
    "https://docs.google.com/": ("google docs", "google documents"),
    "https://drive.google.com/": ("google drive",),
    "https://mail.google.com/": ("gmail", "google mail"),
    "https://maps.google.com/": ("google maps",),
    "https://meet.google.com/": ("google meet",),
    "https://sheets.google.com/": ("google sheets",),
    "https://slides.google.com/": ("google slides",),
    "https://www.youtube.com/": ("youtube", "youtube web"),
    "https://github.com/": ("github", "github web"),
    "https://gitlab.com/": ("gitlab", "gitlab web"),
    "https://app.slack.com/": ("slack", "slack web", "slack web app"),
    "https://discord.com/app": ("discord", "discord web", "discord web app"),
    "https://teams.microsoft.com/": ("microsoft teams", "teams web"),
    "https://outlook.office.com/mail/": ("outlook", "microsoft outlook", "outlook web"),
    "https://www.office.com/": ("microsoft 365", "office 365", "office web"),
    "https://www.notion.so/": ("notion", "notion web", "notion web app"),
    "https://www.figma.com/": ("figma", "figma web", "figma web app"),
    "https://web.whatsapp.com/": ("whatsapp", "whatsapp web"),
    "https://web.telegram.org/": ("telegram", "telegram web"),
    "https://open.spotify.com/": ("spotify", "spotify web", "spotify web player"),
}


def _normalize_web_app_name(value: str) -> str:
    """Normalize spacing and punctuation in a human-facing web-app name."""
    return "".join(character for character in value.casefold() if character.isalnum())


_WEB_APP_ALIASES = {
    _normalize_web_app_name(alias): url
    for url, aliases in _WEB_APPS.items()
    for alias in aliases
}


def _resolve_web_app(query: str) -> str | None:
    """Resolve a known, unambiguous web-app name to its canonical URL."""
    return _WEB_APP_ALIASES.get(_normalize_web_app_name(query))


def open_browser(query: str) -> str:
    """
    Open the default web browser to a specific URL or search query.
    
    This function analyzes the provided query string to determine if it is a 
    direct URL (e.g., starting with http/https), a domain name (e.g., 'example.com'),
    or a plain text search query. It then formulates the appropriate URL and attempts
    to open it using the system's default web browser.
    
    Args:
        query (str): The URL to open or a search term.
    
    Returns:
        str: A message indicating success or failure of the browser launch operation.
    """
    # Validate the input to ensure a query is provided
    if not query or not str(query).strip():
        return "Error: No query provided."
    query = str(query).strip()
    if len(query) > 4096 or any(ord(char) < 32 for char in query):
        return "Error: Invalid browser query."

    # Determine if the query is a direct URL, a known web app, or a search.
    parsed = urllib.parse.urlsplit(query)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        # The query is already a fully formed URL
        url = query
    elif re.fullmatch(r"(?:localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63})(?::\d{1,5})?(?:/[^\s]*)?", query):
        # Simple heuristic for domains like "youtube.com" (contains dot, no spaces)
        # Prefix with https:// to form a valid URL
        url = f"https://{query}"
    elif web_app_url := _resolve_web_app(query):
        url = web_app_url
    else:
        # The query appears to be a search term; format it as a Google search URL
        # URL encode the query to handle spaces and special characters properly
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/?q={encoded_query}"
        
    try:
        # Attempt to open the formulated URL in the default web browser
        # webbrowser.open returns a boolean indicating success
        success = webbrowser.open(url)
        if success:
            return f"Successfully opened browser with URL: {url}"
        else:
            return f"Failed to open browser with URL: {url}"
    except Exception as e:
        # Catch and return any unexpected exceptions during the browser launch
        return f"Error opening browser: {str(e)}"
