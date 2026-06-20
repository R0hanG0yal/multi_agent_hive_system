from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
import re
from time import time
from typing import Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class LiveKnowledge:
    topic: str
    title: str
    summary: str
    source_name: str
    url: str


class KnowledgeRetriever(Protocol):
    def lookup(self, topic: str, domain: str) -> LiveKnowledge | None:
        ...

    def lookup_health(self, topic: str) -> LiveKnowledge | None:
        ...


class InternetKnowledgeRetriever:
    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 60 * 60 * 12,
        timeout_seconds: float = 2.5,
        tool: str = "hackathon_ai_env",
        email: str = "",
    ) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.tool = tool
        self.email = email
        self._cache: dict[tuple[str, str], tuple[float, LiveKnowledge | None]] = {}

    def lookup(self, topic: str, domain: str) -> LiveKnowledge | None:
        cleaned = _clean_topic(topic)
        if not cleaned:
            return None
        cache_key = (f"{domain}:topic", cleaned.lower())
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        result = self._lookup_wikipedia(cleaned)
        self._cache[cache_key] = (time(), result)
        return result

    def lookup_health(self, topic: str) -> LiveKnowledge | None:
        cleaned = _clean_topic(topic)
        if not cleaned:
            return None
        cache_key = ("health:topic", cleaned.lower())
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        result = self._lookup_medlineplus(cleaned)
        self._cache[cache_key] = (time(), result)
        return result

    def _from_cache(self, key: tuple[str, str]) -> LiveKnowledge | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        created_at, value = cached
        if (time() - created_at) > self.cache_ttl_seconds:
            del self._cache[key]
            return None
        return value

    def _request_json(self, url: str) -> dict[str, object] | None:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "HackathonAIEnv/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    def _request_xml(self, url: str) -> ET.Element | None:
        request = Request(
            url,
            headers={
                "Accept": "application/xml,text/xml",
                "User-Agent": "HackathonAIEnv/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return ET.fromstring(response.read())
        except Exception:
            return None

    def _lookup_wikipedia(self, topic: str) -> LiveKnowledge | None:
        search_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": 1,
                "format": "json",
            }
        )
        search_payload = self._request_json(search_url)
        if not search_payload:
            return None
        search_query = search_payload.get("query", {})
        if not isinstance(search_query, dict):
            return None
        search_items = search_query.get("search", [])
        if not isinstance(search_items, list) or not search_items:
            return None
        first_item = search_items[0]
        if not isinstance(first_item, dict):
            return None
        title = str(first_item.get("title", "")).strip()
        if not title:
            return None

        extract_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "redirects": 1,
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        extract_payload = self._request_json(extract_url)
        if not extract_payload:
            return None
        query_payload = extract_payload.get("query", {})
        if not isinstance(query_payload, dict):
            return None
        pages = query_payload.get("pages", [])
        if not isinstance(pages, list) or not pages:
            return None
        page = pages[0]
        if not isinstance(page, dict):
            return None
        summary = _trim_summary(str(page.get("extract", "")))
        if not summary:
            return None
        page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        return LiveKnowledge(
            topic=topic,
            title=title,
            summary=summary,
            source_name="Wikipedia",
            url=page_url,
        )

    def _lookup_medlineplus(self, topic: str) -> LiveKnowledge | None:
        parameters = {
            "db": "healthTopics",
            "term": topic,
            "retmax": 1,
            "rettype": "brief",
            "tool": self.tool,
        }
        if self.email:
            parameters["email"] = self.email
        search_url = "https://wsearch.nlm.nih.gov/ws/query?" + urlencode(parameters)
        root = self._request_xml(search_url)
        if root is None:
            return None
        document = root.find(".//document")
        if document is None:
            return None

        title = ""
        summary = ""
        for content in document.findall("content"):
            name = str(content.attrib.get("name", "")).lower()
            content_text = _trim_summary("".join(content.itertext()))
            if not content_text:
                continue
            if name == "title" and not title:
                title = content_text
            if name in {"fullsummary", "snippet"} and not summary:
                summary = content_text

        if not title and not summary:
            return None
        return LiveKnowledge(
            topic=topic,
            title=title or topic.title(),
            summary=summary or f"MedlinePlus has a health topic page about {topic}.",
            source_name="MedlinePlus",
            url=str(document.attrib.get("url", "https://medlineplus.gov/")).strip(),
        )


def _clean_topic(topic: str) -> str:
    cleaned = re.sub(r"\s+", " ", topic).strip(" .,:;!?")
    return cleaned[:80]


def _trim_summary(text: str) -> str:
    cleaned = unescape(re.sub(r"\s+", " ", text)).strip()
    if len(cleaned) <= 420:
        return cleaned
    return cleaned[:417].rsplit(" ", 1)[0] + "..."
