"""
Tools available to agents in this system.

Each tool is decorated with @tool so LangChain/LangGraph can bind them to LLMs
via llm.bind_tools([...]) and invoke them automatically during ReAct loops.
"""

import json
import urllib.request
import urllib.parse
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# Web search tool (uses DuckDuckGo Instant Answer API — no key required)
# In production swap this for Tavily or SerpAPI for richer results.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for information about a company, role, or topic.
    Returns a plain-text summary of the top result.

    Args:
        query: The search query string.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JobAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        # Prefer AbstractText (Wikipedia-style summary), fall back to RelatedTopics
        if data.get("AbstractText"):
            return data["AbstractText"][:1500]

        topics = data.get("RelatedTopics", [])
        snippets = []
        for t in topics[:5]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"])
        if snippets:
            return "\n\n".join(snippets)[:1500]

        return f"No direct summary found for: {query}. Try a more specific query."

    except Exception as e:
        return f"Search failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# URL fetch tool — reads a webpage and returns clean text
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_url(url: str) -> str:
    """
    Fetch the text content of a webpage URL.
    Useful for reading company About pages, job postings, LinkedIn pages, etc.

    Args:
        url: Full URL including https://
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; JobAgent/1.0)"
                )
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode(errors="replace")

        # Strip HTML tags with a simple regex-free approach
        import re
        text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]

    except Exception as e:
        return f"Could not fetch URL: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# File export tool — writes the final output bundle to disk
# ─────────────────────────────────────────────────────────────────────────────

@tool
def export_to_file(content: str, filename: str) -> str:
    """
    Save a generated document (resume, cover letter, interview prep) to a file.

    Args:
        content:  The text content to save.
        filename: Target filename (e.g. 'cover_letter.md').
    """
    try:
        path = f"/tmp/{filename}"
        with open(path, "w") as f:
            f.write(content)
        return f"Saved to {path}"
    except Exception as e:
        return f"Export failed: {str(e)}"


# All tools exported for easy import elsewhere
ALL_TOOLS = [web_search, fetch_url, export_to_file]
RESEARCH_TOOLS = [web_search, fetch_url]
