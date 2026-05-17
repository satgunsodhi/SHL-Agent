# SHL Conversational Assessment Agent — Engineering Briefing

**Stack:** Python · FastAPI · OpenRouter (DeepSeek V4 Flash) · BM25 · Pydantic · rank_bm25
**Catalog:** 377 SHL assessments · Deployment Target: AWS Lambda

---

## Executive Approach

The assignment required building a conversational agent that recommends psychometric assessments from SHL's product catalog in response to natural-language hiring queries. The solution adopted a **Retrieve-then-Generate (RAG)** architecture: a structured retrieval layer grounds every LLM response in catalog-verified items, preventing hallucination of non-existent products. The agent operates as a **multi-turn dialogue state machine** — `CLARIFY → RETRIEVE → RECOMMEND → REFINE → COMPARE` — maintaining full conversation history per request. The core design principle was that the LLM acts as a **reasoner and ranker**, not a knowledge source; every recommended assessment name and URL must exist in the indexed catalog. A key insight from the development process was that **retrieval quality, not LLM capability, is the binding ceiling on recommendation accuracy.**

---

## Design Choices & Evolution

The initial v1 architecture was built around **Gemini 2.0 Flash** via `google-genai` SDK, with a fully stateless `/chat` endpoint that replayed the full conversation history on each call. This stateless design remains unchanged in the final system — zero server-side session state ensures clean horizontal scaling and no consistency issues across cold starts.

The **LLM backend evolved through three stages** driven by infrastructure constraints:

| Stage | Model | Reason for Change |
|---|---|---|
| v1 | Gemini 2.0 Flash (`google-genai`) | Initial design; no API key available for evaluation |
| v2 | OpenRouter `openai/gpt-oss-120b:free` | Switched for OpenRouter compatibility; too slow on large prompts |
| v3 (final) | OpenRouter `deepseek/deepseek-v4-flash:free` | 1.05M token context, low latency, handles full-catalog injection |

**FastAPI** served the `/chat` and `/health` endpoints. **Pydantic** models enforced strict I/O schemas matching the assignment specification — `reply` (str), `recommendations` (null or array of ≤10 items with `name`, `url`, `test_type`), `end_of_conversation` (bool). The agent's dialogue state machine used the following routing logic:

- **CLARIFY:** Triggered on vague queries; extracts `job_level`, `role_family`, `use_case` (selection vs. development), and constraints before any shortlist is generated. Outputs `recommendations: null`.
- **RECOMMEND:** Constructs a structured shortlist from the retriever's context. Capped at 10 items per response.
- **REFINE:** On follow-up turns, updates the shortlist in-place without discarding prior context.
- **COMPARE:** Generates a side-by-side structured comparison using only catalog-sourced metadata.

---

## Retrieval Setup

**Data Ingestion:** Catalog data was compiled via a custom browser-side JavaScript deep-scraper that paginated through `shl.com/products/product-catalog`, fetching per-product detail pages concurrently. The scraped CSV was merged with an official JSON catalog (`shl_product_catalog.json`) in `catalog_loader.py`, producing 377 canonical `CatalogItem` objects. Each item carries: `entity_id`, `name`, `url`, `test_type_labels`, `duration`, `description`, `job_levels`, `languages`, `remote_testing`, and `adaptive_irt`. A composite `search_text` field was synthesized per item by concatenating `name | description | test_type_labels | job_levels | flags` to maximise retrieval signal.

**Phase 1 — Dense Retrieval (v1):** `sentence-transformers/all-MiniLM-L6-v2` (384-dim) embedded all items into a FAISS `IndexFlatIP` (cosine similarity). Initial `top_k=15` was later raised to `top_k=75` to surface foundational assessments like **OPQ32r**, which ranked 61st for specific job-role queries due to its intentionally generic description.

**Phase 2 — Hybrid Retrieval (v2):** `rank_bm25.BM25Okapi` was added alongside FAISS for lexical matching. BM25 excelled where dense embeddings failed: exact product-name disambiguation (e.g., `Sales Transformation Report` v1 vs. v2). A **CrossEncoder re-ranker** (`ms-marco-MiniLM-L-6-v2`) rescored the merged candidate list for a final ranking.

**Phase 3 — BM25-Only (Final, Lambda-Compliant):** The PyTorch + `sentence-transformers` stack exceeded the **500 MB Lambda ephemeral storage limit**. The entire dense retrieval pipeline was removed. The final system runs **BM25-only retrieval** augmented by **Core Item Injection** — seven foundational assessments (OPQ32r, Verify G+, GSA, Graduate Scenarios, DSI, Executive Scenarios, Management Scenarios) are **always prepended** to the retrieval result list regardless of query specificity, guaranteeing they appear in the LLM context window.

An upstream **LLM query-rewriting call** rewrote the raw multi-turn conversation history into a single keyword-dense search string before BM25 scoring — reducing noise from conversational filler tokens.

---

## Prompt Design

The system prompt followed a six-section architecture:

1. **Role Definition** — positions the agent as an SHL assessment specialist; explicitly scoped to catalog-only recommendations.
2. **Behavioral Rules** — explicit CLARIFY / RECOMMEND / REFINE / COMPARE routing logic with worked examples.
3. **Scope Guard** — hard refusal of off-topic queries and prompt injection. Pre-LLM regex filters catch obvious injection patterns before the prompt is constructed.
4. **Output Format** — JSON schema enforced verbatim in the prompt. Field-level rules governed `recommendations: null` vs. array, and `end_of_conversation` semantics.
5. **Catalog Context** — a **two-tier injection**: (a) full rich descriptions for top-k BM25 candidates, and (b) a compact `name | URL | duration` table of **all 377 catalog items** as a reference grounding layer.
6. **Conversation History** — full User/Assistant message log formatted as structured pairs.

**Temperature** was set to `0.3` to balance factual grounding with natural phrasing. **Chain-of-Thought (CoT) framing** in the CLARIFY state instructed the model to reason through `[job_level → role_family → use_case → constraints]` before formulating a single, high-information clarifying question. **Version disambiguation rules** were injected after observing persistent product confusion: *"When multiple products share similar names, prefer the exact URL slug match; prefer newer versioned variants (e.g., `salestransformationreport2-0`) unless the user specifies otherwise."* Post-generation, all recommendation URLs were validated against `retriever.url_set`; hallucinated items were silently dropped or corrected via name-based BM25 lookup.

---

## What Did Not Work — Lessons Learned

- **Pure semantic search on foundational assessments:** Dense embeddings ranked OPQ32r at **position 61** for specific role queries. Generic-description items are semantically dissimilar from specific job-role queries by design — a structural mismatch that pure cosine similarity cannot resolve.
- **Direct LLM catalog search (no retrieval layer):** Without a retrieval grounding step, Gemini hallucinated assessment names and URLs not present in the catalog. The RAG layer with URL post-validation was non-negotiable.
- **Brute-force `top_k=75` expansion:** Temporarily raised to expose foundational assessments; this pushed per-request prompt size to ~40K tokens, causing **consistent timeout failures** on free-tier OpenRouter inference, particularly with `gpt-oss-120b`.
- **Naive URL hallucination correction via substring matching:** `str.contains()` matching to snap hallucinated URLs to catalog entries was brittle and failed on partial-name overlaps. Replaced by BM25 fuzzy-name lookup.
- **PyTorch + sentence-transformers on Lambda:** Combined dependency size (~900 MB) exceeded the 500 MB ephemeral storage constraint, requiring the full dense retrieval stack to be stripped. Final deployable footprint: < 150 MB.
- **High LLM temperature (0.7+):** Early experiments with relaxed temperature produced natural but factually loose output, including invented test names. Clamped to `0.3` for production.
- **Single-turn retrieval query:** Using only the latest user message for BM25 scoring lost critical context from earlier turns. Multi-turn query synthesis via the upstream rewriting call was essential.

---

## Evaluation Method & Measuring Improvement

**Programmatic Evaluation (`evaluate.py`):** All 10 annotated conversation traces (`C1.md`–`C10.md`) were replayed via FastAPI's `TestClient`. Each user turn was submitted sequentially; the agent's final `recommendations[].url` slugs were compared against the ground-truth expected set per trace.

**Primary Metric — Recall@10:** Defined as the fraction of ground-truth assessments present in the agent's top-10 output.

$$\text{Recall@10} = \frac{|\text{recommended} \cap \text{expected}|}{|\text{expected}|}$$

**Secondary Metric — Behavior Probes (5 pass/fail checks):** Clarification protocol on vague queries; ≤10 item output constraint; zero hallucinated URLs; shortlist updates on refinement requests; structured output on comparison queries.

**Recall@10 Progression:**

| Version | Recall@10 | Key Change |
|---|---|---|
| v1 — FAISS only, `top_k=15` | **22.8%** | Baseline dense retrieval |
| v2 — `top_k=75` expansion | **~45%** | Wider retrieval window; timeout issues |
| v3 — Hybrid BM25 + core injection + full catalog | **56.2%** | Two-tier context, foundational item pinning |
| v4 — Disambiguation prompt rules added | **~65%** | Version-resolution instructions |

**Per-Trace Highlights:** C9 (graduate management trainee) improved from **0% → 100%** after core item injection. C10 (graduate cognitive + scenarios) improved from **50% → 100%** in v3. Hardest remaining cases were C3 (contact center specialist variants), C7 (medical/administrative roles), and C8 (Microsoft 365 skill version disambiguation), all suffering from fine-grained product-name similarity problems the prompt disambiguation rules only partially resolved.

---

## AI Tools Used

- **OpenRouter / DeepSeek V4 Flash** as the core reasoning LLM in the final production agent.
- **GitHub Copilot (Antigravity)** for scaffolding, boilerplate generation, and iterative prompt refinement.
- All architectural decisions, retrieval strategy choices, evaluation methodology, and prompt engineering were developed with deliberate understanding of the trade-offs involved.
