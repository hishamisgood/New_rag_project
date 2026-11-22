"""Optional external search via Tavily or Serper's HTTP APIs."""

import logging
import os

import requests
from langchain_core.documents import Document

from .sql_tool import ToolResult

logger = logging.getLogger(__name__)


def _to_documents(items: list[dict], text_key: str, url_key: str, max_results: int) -> list[Document]:
    return [
        Document(
            page_content=str(item.get(text_key, "")),
            metadata={
                "source": f"web:{item.get(url_key, '')}",
                "title": str(item.get("title", "")),
            },
        )
        for item in items[:max_results]
        if item.get(url_key) and item.get(text_key)
    ]


class WebSearchTool:
    """Fetch public web results without requiring an extra SDK."""

    name = "web_search"

    def run(self, question: str, max_results: int = 4) -> ToolResult:
        tavily_key = os.getenv("TAVILY_API_KEY")
        serper_key = os.getenv("SERPER_API_KEY")
        if not tavily_key and not serper_key:
            logger.warning("Web search skipped because no Tavily or Serper API key is set")
            return ToolResult(self.name, [], None)

        try:
            if tavily_key:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": tavily_key, "query": question, "max_results": max_results},
                    timeout=15,
                )
                response.raise_for_status()
                results = response.json().get("results", [])[:max_results]
                documents = _to_documents(results, "content", "url", max_results)
            else:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": question, "num": max_results},
                    timeout=15,
                )
                response.raise_for_status()
                results = response.json().get("organic", [])[:max_results]
                documents = _to_documents(results, "snippet", "link", max_results)
            return ToolResult(self.name, documents, None)
        except Exception as exc:
            logger.warning("Web search request failed: %s", exc)
            return ToolResult(self.name, [], None)
