"""A chat agent with unbounded memory in ~25 lines. Runs offline with a stub "LLM".

Swap `fake_llm` for any model call; the memory pattern stays identical:
    1. recall   -> inject a small, budgeted block of relevant memories
    2. respond  -> your LLM
    3. remember -> store the exchange
    4. sleep    -> now and then, let old memories settle
"""
from sediment import Sediment


def fake_llm(prompt: str) -> str:
    seen = [l for l in prompt.splitlines() if l.startswith("- (")]
    return f"(I recall {len(seen)} relevant memories) Got it."


def main() -> None:
    mem = Sediment("agent_memory.db")
    mem.remember("The user's name is Sam and Sam prefers concise answers.", pin=True)

    for turn, user in enumerate(
        [
            "Remember that our staging database is called orion-stg.",
            "What did I say the staging database was called?",
            "Which name do you use for me?",
        ]
    ):
        context = mem.context(user, budget=400)
        reply = fake_llm(f"MEMORY\n{context}\n\nUSER: {user}")
        print(f"you   > {user}\nagent > {reply}\n{context}\n")
        mem.remember(f"User said: {user}")
        if turn % 50 == 49:
            mem.sleep()
    mem.close()


if __name__ == "__main__":
    main()
