# Phase 8 Completion Report

**Status:** complete. 836 tests pass in the repository (183 new for Phase 8); `ruff check .` is clean.

8.1 four memory categories · 8.2 storage abstraction, SQLite (default), persistent vector index, optional Qdrant
adapter (tested against the real client in local mode), dedupe, update/delete · 8.3 extraction from messages,
summaries, tool results, workflows and explicit preferences with importance scoring, classification, metadata,
retention · 8.4 keyword + semantic retrieval, recency/importance weighting, metadata filters, context ranking ·
8.5 eight-node LangGraph integration wrapping Agent or Orchestrator · 8.6 conversation ids, history, summaries,
context-window budgeting, retrieval, archival · 8.7 add/retrieve/update/delete/clear/retention/sensitive-data
exclusion · 8.8 tests for every listed area, plus failure injection.

## Fixes found while testing
* Card-number redaction swallowed the following space; phone pattern mangled 16-digit numbers.
* In-memory SQLite shared one connection across worker threads (intermittent errors under concurrency); access is
  now serialized. The same latent issue existed in the Phase 5 registry store and was fixed there too.
* Message timestamps ignored the injected clock; `append_message` now takes the caller's timestamp.
* Context windows now report `truncated` when the hard fetch cap cuts off history.
* Retrieval noise: the words "user"/"use" (in nearly every memory) are stopwords; only pinned memories and
  behavioural directives are standing, not ordinary taste preferences.

## Known gaps
Semantic search uses a hashing embedder by default (lexical/sub-word, not true synonyms). Extraction rules are
English-only. The LLM extractor/summarizer are verified with scripted models, not live LLMs. No REST/API layer
for the controls yet. `StoredMessage`/memory content is not encrypted at rest. Verified on Python 3.12.
