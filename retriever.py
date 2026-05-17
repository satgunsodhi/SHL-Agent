"""
Retriever — Lightweight BM25-only search over the SHL catalog.
"""

import re
from typing import List, Dict, Set
from rank_bm25 import BM25Okapi

from catalog_loader import CatalogItem, load_catalog

CORE_ASSESSMENT_NAMES = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "Global Skills Assessment",
    "Graduate Scenarios",
    "Dependability and Safety Instrument (DSI)",
    "Executive Scenarios",
    "Management Scenarios",
]

class CatalogRetriever:
    def __init__(self):
        self.catalog = load_catalog()

        self.name_index = {}
        self.url_set = set()
        for item in self.catalog:
            self.name_index[item.name.lower()] = item
            self.url_set.add(item.url)

        self.core_items = []
        for name in CORE_ASSESSMENT_NAMES:
            item = self.name_index.get(name.lower())
            if item:
                self.core_items.append(item)

        tokenized_corpus = [self._tokenize(item.search_text) for item in self.catalog]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._compact_catalog = self._build_compact_catalog()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _bm25_search(self, query: str, top_k: int) -> List[CatalogItem]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        scored_items = [(score, self.catalog[i]) for i, score in enumerate(scores) if score > 0]
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:top_k]]

    def hybrid_retrieve(self, query: str, top_k: int = 30) -> List[CatalogItem]:
        candidates = self._bm25_search(query, top_k=top_k*2)
        
        final_results = []
        final_seen = set()

        def _add_final(item: CatalogItem):
            if item.entity_id not in final_seen:
                final_results.append(item)
                final_seen.add(item.entity_id)

        for item in self.core_items:
            _add_final(item)

        for item in candidates:
            if len(final_results) >= top_k + len(self.core_items):
                break
            _add_final(item)

        return final_results

    def find_items_by_names(self, names: List[str]) -> List[CatalogItem]:
        items = []
        for n in names:
            item = self.name_index.get(n.lower())
            if item:
                items.append(item)
        return items

    def _build_compact_catalog(self) -> str:
        lines = ["=== FULL SHL CATALOG REFERENCE ==="]
        for item in self.catalog:
            lines.append(f"{item.name} | {item.url} | {item.duration}")
        return '\n'.join(lines)

    def get_compact_catalog(self) -> str:
        return self._compact_catalog

    def get_catalog_context(self, items: List[CatalogItem]) -> str:
        details = []
        for item in items:
            t = f"Name: {item.name}\nURL: {item.url}\nDescription: {item.description}\nTypes: {item.test_type_labels}\nDuration: {item.duration}"
            details.append(t)
        return '\n\n'.join(details)


_retriever_instance = None
def get_retriever() -> CatalogRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = CatalogRetriever()
    return _retriever_instance