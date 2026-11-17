"""Time-lapse demo: 150 days in the life of a coding assistant, simulated in a second."""
from __future__ import annotations

import random
import time

from .core import DAY, Sediment

THINGS = ["auth", "billing", "search", "export", "cache", "queue", "onboarding", "metrics"]
PEOPLE = ["Maya", "Ravi", "Dana", "Leo"]
PROJECTS = ["Atlas", "Nimbus", "Harbor"]
TEMPLATES = [
    "Met with {p} about {j}. We agreed to ship the {t} work by Friday.",
    "{j} build failed on CI because of a flaky {t} test. Fixed it by pinning the dependency.",
    "Reviewed {p}'s pull request for {j}; suggested renaming the {t} module.",
    "Debugged a memory leak in the {j} {t} worker. Root cause was an unclosed connection.",
    "{p} prefers async updates over standups for {j}.",
    "Refactored {t} in {j}. Latency dropped after batching the requests.",
]
PLANTED = {
    2: ("Decision: Atlas will use PostgreSQL 16 with pgvector for embeddings, because Maya benchmarked it against Qdrant.", {}),
    4: ("Ravi's birthday is March 14. He loves ube cheesecake.", {}),
    0: ("The user's name is Sam. Sam wants concise answers with code first.", {"pin": True}),
}

QUESTIONS = [
    "Why did we pick pgvector for Atlas?",
    "When is Ravi's birthday and what does he love?",
    "What is the user's name?",
]


def run(days: int = 150, seed: int = 7, quiet: bool = False) -> Sediment:
    from .cli import render_strata  # local import: cli imports this module

    rng = random.Random(seed)
    t = [time.time() - days * DAY]
    m = Sediment(":memory:", clock=lambda: t[0])
    say = (lambda *a: None) if quiet else print

    say(f"\n  simulating {days} days of agent life…\n")
    for day in range(days):
        t[0] = time.time() - (days - day) * DAY
        if day in PLANTED:
            text, kw = PLANTED[day]
            m.remember(text, **kw)
        for _ in range(rng.randint(2, 4)):
            m.remember(rng.choice(TEMPLATES).format(p=rng.choice(PEOPLE), j=rng.choice(PROJECTS), t=rng.choice(THINGS)))
        if day % 7 == 6:
            m.sleep()
    t[0] = time.time()

    say(render_strata(m))
    say("\n  ── asking questions about things said months ago ─────────────────────")
    for q in QUESTIONS:
        say(f"\n  ❯ {q}")
        for line in m.recall(q, budget=260).render(header=False).splitlines():
            say("    " + line)
    say("")
    return m


if __name__ == "__main__":
    run()
