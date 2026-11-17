"""Use an LLM to write the gists instead of the built-in extractive summarizer.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python examples/llm_summarizer.py

Any callable with the signature (list_of_texts, max_chars) -> str works.
"""
import os

from sediment import Sediment


def claude_summarizer(texts, max_chars):
    import anthropic  # imported lazily: Sediment itself has zero dependencies

    client = anthropic.Anthropic()
    notes = "\n".join(f"- {t}" for t in texts)
    msg = client.messages.create(
        model=os.environ.get("SEDIMENT_MODEL", "claude-haiku-5-5"),
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Condense these memories into at most {max_chars} characters. "
                       f"Keep names, decisions, numbers and dates. Reply with the summary only.\n\n{notes}",
        }],
    )
    return msg.content[0].text.strip()


if __name__ == "__main__":
    mem = Sediment("llm_memory.db", summarizer=claude_summarizer)
    mem.remember("Decision: Atlas uses PostgreSQL 16 with pgvector.")
    mem.sleep(threshold=1.1)  # force compaction just to see the summarizer run
    print(mem.recall("which database for Atlas?"))
