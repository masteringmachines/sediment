# Sediment

**Infinite memory for AI agents. Memories don't get deleted, they settle.**

Zero dependencies. One SQLite file. Under 800 lines of Python.

```
  SEDIMENT · core sample
  ──────────────────────────────────────────────────────────────────
  ░ fresh   ██······················    44  recent episodes, full detail
  ▒ gist    ██······················    33  faded → compact summaries
  ▓ bedrock █·······················    30  long-term themes
  █ fossils ████████████████████████   489  raw originals, zlib, never deleted
```

## The idea

Most agent memory is either a context window that overflows or a vector store that
grows into a haystack. Sediment copies how geology (and brains) do it:

```
 remember ──▶ FRESH ──fades──▶ GIST ──fades──▶ BEDROCK
                │                │                │
                └────────────────┴────────────────┴──▶ FOSSILS
                       every original, compressed, never deleted
```

- **Forgetting curve.** Every memory has a retention score that decays over time
  (Ebbinghaus-style). Important and pinned memories decay slower or not at all.
- **Spaced repetition.** Recalling a memory makes it stick longer, and recalling one that was
  almost forgotten helps more.
- **Sleep.** `sleep()` finds the faded memories, clusters them by topic and entities, and
  compacts each cluster into a gist. Gists can later settle into bedrock themes.
- **Fossils.** The originals are zlib-compressed into cold storage, not deleted. A gist keeps a
  *scent*, the keywords of everything it absorbed. A query can therefore still find a gist
  even after the summary dropped the detail, and recall then **digs** the matching sentence
  out of the fossils.
- **Association.** Memories that share rare names, tags or projects prime each other
  during recall.

"Infinite" here means storage grows without bound while the cost of a recall stays bounded:
you give `recall()` a token budget and it packs the best memories into it.

## Try it in 10 seconds

```bash
git clone https://github.com/<you>/sediment && cd sediment
python -m sediment demo
```

The demo simulates 150 days of an agent's life. A decision made on day 2 gets compacted into
a gist, and recall still finds it months later:

```
  ❯ Why did we pick pgvector for Atlas?
    - (gist, 4mo ago) Decision: Atlas will use PostgreSQL 16 with pgvector for embeddings,
      because Maya benchmarked it against Qdrant.
```

## Use it

```python
from sediment import Sediment

mem = Sediment("agent.db")                       # or ":memory:"
mem.remember("The user's name is Sam.", pin=True)  # pinned: never fades
mem.remember("Decision: Atlas uses pgvector.", tags=["atlas"])

print(mem.context("which database for Atlas?", budget=400))  # drop this into your prompt
# [sediment: 1 memories, ~10 tokens]
# - (today) Decision: Atlas uses pgvector.

mem.sleep()           # run nightly, or every N turns
mem.dig(gist_id)      # get the unabridged originals under any gist
mem.erase(memory_id)  # hard delete, including every fossil beneath it
```

An agent loop is in [`examples/agent_loop.py`](examples/agent_loop.py).

### Bring your own summarizer

The default summarizer is extractive and offline. To have an LLM write the gists, pass any
`(texts, max_chars) -> str`:

```python
mem = Sediment("agent.db", summarizer=my_llm_summarizer)
```

See [`examples/llm_summarizer.py`](examples/llm_summarizer.py) for a Claude version.

### CLI

```bash
sediment remember "Ravi's birthday is March 14" --tag people
sediment recall "when is Ravi's birthday"
sediment sleep
sediment strata          # the core sample above
sediment dig 42          # originals beneath a gist
```

The database path comes from `--db` or `$SEDIMENT_DB` (default `./sediment.db`).

## How recall scores

`0.62 × relevance (BM25 over text, tags and scent) + 0.23 × how well it is still remembered
+ 0.15 × association`. Weak matches are dropped relative to the best one, then the rest are
packed greedily into your token budget.

## Honest limits

- Retrieval is lexical (SQLite FTS5 with Porter stemming), not semantic. The trade-off is
  zero dependencies, no model downloads, and fully inspectable results. Synonyms won't match.
- The default summarizer is extractive. It picks sentences and doesn't rewrite them. Use an
  LLM summarizer for fluent gists.
- Token counts are estimated at about 4 characters per token.
- One writer at a time (SQLite default locking).

## Develop

```bash
python -m unittest discover -s tests -v
```

Requires Python 3.9+ with SQLite FTS5 (included in nearly every build, and there is a slower
fallback if it's missing). MIT licensed.
