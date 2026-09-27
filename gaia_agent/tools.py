import requests
import trafilatura
from ddgs import DDGS


def web_search(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=5))
    except Exception as e:
        return f"ERROR: web search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}\n  {r.get('href')}\n  {r.get('body')}")
    return "\n".join(lines)


WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via DuckDuckGo and return the top results (title, URL, snippet).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
}


MAX_PAGE_CHARS = 8000


def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: could not fetch {url}: {e}"
    text = trafilatura.extract(resp.text, with_metadata=True) or ""
    if not text.strip():
        text = resp.text
    return text[:MAX_PAGE_CHARS]


FETCH_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "Download a web page and return its main text content (HTML stripped).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}
            },
            "required": ["url"],
        },
    },
}
