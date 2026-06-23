"""
tools/browser.py - Browser control tool.

This module provides a unified tool for opening URLs and performing web searches
in the user's default web browser.
"""
import urllib.parse
import webbrowser

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
    if not query:
        return "Error: No query provided."

    # Determine if the query is a direct URL or requires formatting
    if query.startswith("http://") or query.startswith("https://"):
        # The query is already a fully formed URL
        url = query
    elif "." in query and " " not in query:
        # Simple heuristic for domains like "youtube.com" (contains dot, no spaces)
        # Prefix with https:// to form a valid URL
        url = f"https://{query}"
    else:
        # The query appears to be a search term; format it as a Google search URL
        # URL encode the query to handle spaces and special characters properly
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
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
