"""
Retriever — Hybrid semantic + keyword search over the SHL catalog.

Strategy:
  1. Semantic search (FAISS + MiniLM) for broad intent matching
  2. Keyword/substring search on product names for explicit mentions
  3. Always-include set of foundational cross-cutting assessments
  4. Full catalog compact reference table for the LLM to browse

This hybrid approach ensures that:
  - Specific product mentions (e.g. "OPQ32r", "Verify G+") are always found
  - Semantic intent (e.g. "leadership assessment") surfaces relevant items
  - Foundational assessments (OPQ32r, Verify G+) are always in context
  - The LLM has a complete catalog reference to draw from
"""

import os
import re
import numpy as np
from typing import List, Optional, Dict, Any, Set, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

from catalog_loader import CatalogItem, load_catalog


# ── Foundational assessments that appear in most shortlists ─────────────────
# These are cross-cutting instruments useful for almost every hiring scenario.
# They rank poorly in semantic search because their descriptions are generic.
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
    """
    Hybrid retrieval engine combining:
      - Dense semantic search (FAISS + sentence-transformers)
      - Sparse keyword matching on product names & descriptions
      - Always-include foundational assessment set
      - Full catalog compact reference for LLM browsing
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", lazy_models: bool = True):
        self.model_name = model_name
        self._lazy_models = lazy_models
        self._models_ready = False
        self.model = None
        self.cross_encoder = None
        self.embeddings = None
        self.index = None
        self.bm25 = None

        print("📂 Loading catalog...")
        self.catalog: List[CatalogItem] = load_catalog()

        # Build indexes
        self.name_index: Dict[str, CatalogItem] = {}
        self.url_set: Set[str] = set()
        for item in self.catalog:
            self.name_index[item.name.lower()] = item
            self.url_set.add(item.url)

        # Identify core/foundational assessments by name
        self.core_items: List[CatalogItem] = []
        for name in CORE_ASSESSMENT_NAMES:
            item = self.name_index.get(name.lower())
            if item:
                self.core_items.append(item)

        # Pre-build the compact full catalog reference
        self._compact_catalog = self._build_compact_catalog()

        if not self._lazy_models:
            self._ensure_models()

    def _ensure_models(self) -> None:
        if self._models_ready:
            return

        print("🔧 Loading embedding model...")
        self.model = SentenceTransformer(self.model_name)

        print("🔧 Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("🧮 Computing embeddings...")
        search_texts = [item.search_text for item in self.catalog]
        self.embeddings = self.model.encode(
            search_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

        # Build FAISS index (inner product on L2-normalized = cosine sim)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        print("🔠 Building BM25 index...")
        tokenized_corpus = [self._tokenize(item.search_text) for item in self.catalog]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._models_ready = True

        print(f"✅ Retriever ready: {len(self.catalog)} items indexed "
              f"(dim={dim}, {len(self.core_items)} core items)")

    # ── Semantic search ─────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[CatalogItem, float]]:
        """Return top-K items by cosine similarity with scores."""
        self._ensure_models()
        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        search_k = min(top_k, len(self.catalog))
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.catalog[idx], float(score)))
        return results

    # ── Keyword search ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _bm25_search(self, query: str, top_k: int) -> List[CatalogItem]:
        """Lexical search using BM25."""
        self._ensure_models()
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        return [self.catalog[i] for i in top_n if scores[i] > 0]

    # ── Hybrid retrieve ─────────────────────────────────────────────────────

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[CatalogItem]:
        """
        Combine multiple retrieval strategies for maximum recall:
          1. Fetch broad candidate pool via dense semantic + BM25 sparse search
          2. Rerank candidates with CrossEncoder
          3. Prepend core/foundational assessments
        Deduplicated and returned as a single list.
        """
        self._ensure_models()
        seen_ids: Set[str] = set()
        candidates: List[CatalogItem] = []

        def _add(item: CatalogItem):
            if item.entity_id not in seen_ids:
                candidates.append(item)
                seen_ids.add(item.entity_id)

        # 1. Semantic search (broad intent matching)
        semantic_results = self._semantic_search(query, top_k=100)
        for item, score in semantic_results:
            _add(item)

        # 2. Lexical search (exact keyword matching)
        bm25_results = self._bm25_search(query, top_k=50)
        for item in bm25_results:
            _add(item)

        # 3. Rerank candidates using CrossEncoder
        if candidates:
            pairs = [[query, item.search_text] for item in candidates]
            scores = self.cross_encoder.predict(pairs)
            
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [item for item, score in scored_candidates]

        # 4. Final assembly
        final_results: List[CatalogItem] = []
        final_seen: Set[str] = set()

        def _add_final(item: CatalogItem):
            if item.entity_id not in final_seen:
                final_results.append(item)
                final_seen.add(item.entity_id)

        # Core assessments are always prepended
        for item in self.core_items:
            _add_final(item)

        # Append top K from reranked candidates
        for item in candidates:
            if len(final_results) >= top_k + len(self.core_items):
                break
            _add_final(item)

        return final_results

    # ── Legacy interface (for backward compat) ──────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CatalogItem]:
        """Retrieve using hybrid strategy. Delegates to hybrid_retrieve."""
        results = self.hybrid_retrieve(query, top_k=top_k)
        if filters:
            results = [item for item in results if self._passes_filters(item, filters)]
        return results

    def find_by_name(self, name: str) -> Optional[CatalogItem]:
        """Look up a catalog item by exact or fuzzy name match."""
        result = self.name_index.get(name.lower())
        if result:
            return result

        name_lower = name.lower()
        for key, item in self.name_index.items():
            if name_lower in key or key in name_lower:
                return item

        return None

    def find_items_by_names(self, names: List[str]) -> List[CatalogItem]:
        """Find multiple catalog items by name."""
        results = []
        for name in names:
            item = self.find_by_name(name)
            if item:
                results.append(item)
        return results

    # ── Context formatting ──────────────────────────────────────────────────

    def get_catalog_context(self, items: List[CatalogItem]) -> str:
        """Format items as detailed context for the LLM."""
        if not items:
            return "No matching items found in the catalog."

        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"--- Item {i} ---")
            lines.append(f"Name: {item.name}")
            lines.append(f"URL: {item.url}")
            lines.append(f"Test Type: {item.test_type_codes} ({', '.join(item.test_type_labels)})")
            lines.append(f"Description: {item.description}")
            if item.job_levels:
                lines.append(f"Job Levels: {', '.join(item.job_levels)}")
            if item.languages:
                langs = item.languages[:5]
                extra = f" (+{len(item.languages) - 5} more)" if len(item.languages) > 5 else ""
                lines.append(f"Languages: {', '.join(langs)}{extra}")
            if item.duration:
                lines.append(f"Duration: {item.duration}")
            lines.append(f"Remote: {item.remote} | Adaptive: {item.adaptive}")
            lines.append("")

        return "\n".join(lines)

    def _build_compact_catalog(self) -> str:
        """Build a compact one-line-per-item reference of the entire catalog."""
        lines = ["=== FULL SHL CATALOG REFERENCE (use exact names and URLs from this list) ==="]
        for item in self.catalog:
            duration = item.duration if item.duration else "-"
            langs = ", ".join(item.languages[:3]) if item.languages else "-"
            if len(item.languages) > 3:
                langs += f" (+{len(item.languages)-3})"
            lines.append(
                f"{item.name} | {item.test_type_codes} | {duration} | {langs} | {item.url}"
            )
        lines.append("=== END FULL CATALOG ===")
        return "\n".join(lines)

    def get_compact_catalog(self) -> str:
        """Return the pre-built compact full catalog reference."""
        return self._compact_catalog

    # ── Filters ─────────────────────────────────────────────────────────────

    @staticmethod
    def _passes_filters(item: CatalogItem, filters: Dict[str, Any]) -> bool:
        if "job_levels" in filters:
            required_levels = [l.lower() for l in filters["job_levels"]]
            item_levels = [l.lower() for l in item.job_levels]
            if not any(rl in item_levels for rl in required_levels):
                return False
        if "test_type" in filters:
            required_types = [t.strip().upper() for t in filters["test_type"].split(",")]
            item_types = [t.strip().upper() for t in item.test_type_codes.split(",")]
            if not any(rt in item_types for rt in required_types):
                return False
        if "remote" in filters:
            if item.remote.lower() != filters["remote"].lower():
                return False
        if "adaptive" in filters:
            if item.adaptive.lower() != filters["adaptive"].lower():
                return False
        return True


# ── Singleton ───────────────────────────────────────────────────────────────
_retriever_instance: Optional[CatalogRetriever] = None


def get_retriever() -> CatalogRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        eager = os.getenv("EAGER_RETRIEVER", "0") == "1"
        _retriever_instance = CatalogRetriever(lazy_models=not eager)
    return _retriever_instance


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = get_retriever()

    test_queries = [
        "I need a Java developer assessment",
        "Leadership assessment for senior executives",
        "Entry-level contact center screening",
        "Personality test for sales roles",
        "Excel and Word for admin assistants",
        "Senior Rust engineer high-performance networking",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.hybrid_retrieve(query, top_k=20)
        for i, item in enumerate(results[:10], 1):
            print(f"   {i}. {item.name} [{item.test_type_codes}]")
