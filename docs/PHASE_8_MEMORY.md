# Phase 8 — Agent Memory & Context Management

Code: `agent/memory/`. Builds on the existing graphs without modifying them: the memory graph
wraps a Phase 6 `Agent` or Phase 7 `Orchestrator` as its "execute" node.

## Memory categories
| Category | Where | Notes |
|---|---|---|
| Short-term | `short_term.py` | Bounded in-process: recent turns, execution context, intermediate tool results. LRU across conversations. Never persisted. |
| Long-term | `MemoryType.LONG_TERM` | preferences, facts, conversation summaries, context |
| Episodic | `MemoryType.EPISODIC` | tasks, outcomes, decisions, workflow events |
| Semantic | `MemoryType.SEMANTIC` | knowledge and relationships (`subject/predicate/object` metadata) |

## Layout
| Module | Role |
|---|---|
| `models.py`, `settings.py` | Typed records/filters; `MemorySettings`, `RetentionPolicy`, `RetrievalSettings` |
| `store.py` | `MemoryStore`, `ConversationStore`, `VectorIndex` protocols (backend-independent) |
| `sqlite.py` | SQLite stores + persistent vector index (SQLAlchemy Core, worker threads; PostgreSQL-ready) |
| `vector.py` | `InMemoryVectorIndex`, optional `QdrantVectorIndex` (`pip install nexus-mcp[qdrant]`) |
| `embeddings.py` | `Embedder` protocol; `HashingEmbedder` (deterministic, offline) |
| `privacy.py` | Sensitive-data filter, PII redaction, "don't remember" detection |
| `extraction.py` | Classification, importance scoring, rule-based + optional LLM extractor, candidate evaluator |
| `retrieval.py` | Keyword + semantic retrieval, recency/importance weighting, filters, token-budgeted context |
| `service.py` | `MemoryService`: add/get/update/delete, dedupe, retention, summary upsert |
| `conversation.py` | Conversations, messages, context windows, summaries, pruning, archival |
| `graph.py`, `manager.py` | LangGraph integration (`MemoryAgent`) and the `MemoryManager` facade |

## Usage
```python
manager = MemoryManager.sqlite("sqlite:///./nexus_memory.db")          # fully local
agent = MemoryAgent(manager, Orchestrator.create(container, provider))  # or an Agent
result = await agent.run("user-1", "I prefer dark mode", conversation_id=None)
result.response, result.conversation_id, result.persisted, result.retrieved
```
Controls: `manager.memory.add_memory / get_memory / update_memory / delete_memory / retrieve`,
`manager.clear_conversation_memory(user, conv, include_messages=...)`,
`manager.configure_retention(RetentionPolicy(...))`, `manager.apply_retention()`,
`manager.conversations.set_memory_enabled(user, conv, False)` (incognito).

## Graph
`load_context -> [retrieve_memories] -> prepare_context -> run_agent -> [extract_memories -> evaluate_importance
-> [persist_memories]] -> update_conversation`. Bracketed steps are skipped when memory is disabled (globally or
per conversation), the user said "off the record", the run failed, or nothing was approved. Memory failures are
recorded in `memory_errors` and never block the answer.

## Retrieval
`relevance` = weighted keyword overlap + vector similarity (weights renormalize if one signal is missing).
`score = 0.6*relevance + 0.2*recency + 0.2*importance`; recency halves per type half-life
(episodic 21d, long-term 180d, semantic 365d). A memory must clear `min_relevance`, so important-but-unrelated
memories are never injected. Pinned memories and behavioural directives ("always answer in metric") are
*standing* and always included (max 3). Output is deterministic (score, importance, id).
Memory is injected as a clearly labelled **background-data** message, never as instructions.

## Privacy
Blocked (never stored): secrets/API keys/tokens/credentials, private keys, payment cards (Luhn), government IDs,
sensitive-topic terms (health, religion, income, ...; configurable), custom `restricted_terms`. PII (email, phone)
is redacted by default (`allow`/`block` modes available). Persisted messages have secrets redacted. Users are
isolated everywhere (unknown vs foreign ids are indistinguishable). Rejected previews never echo secrets.

## Retention
Explicit TTLs, per-type max age, low-importance ageing, per-type capacity (lowest value evicted first), message
caps (only already-summarized messages are pruned). Pinned memories are exempt. Configurable at runtime.

## Known limits
`HashingEmbedder` captures lexical/sub-word overlap, not true synonyms ("theme" will not find "mode"); plug a neural
`Embedder` in for deeper recall. The SQLite vector index is exact brute force (fine for thousands of memories per
user); use Qdrant beyond that. Rule-based extraction is English-only and deliberately conservative.
