"""Tiny, dependency-free text helpers: tokens, entities, extractive summaries."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Set

STOP = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been
    being it its this that these those i you he she we they me my your our their them his her do
    does did have has had not no so can could will would should may might just about into over
    than too very also what which who whom when where why how there here up out s t""".split()
)

_WORD = re.compile(r"[a-z0-9][a-z0-9_'-]*")
_CAP = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")
_TAGISH = re.compile(r"[#@]\w+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

CUES = (
    "remember", "important", "always", "never", "must", "prefer", "decision", "decided",
    "deadline", "critical", "don't forget", "name is", "birthday", "allerg", "goal",
)


def tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def keywords(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in STOP and len(t) > 1]


def stem(w: str) -> str:
    """Deliberately crude suffix stripper, only used for clustering/overlap."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def kwset(text: str) -> Set[str]:
    return {stem(k) for k in keywords(text)}


def entities(text: str) -> Set[str]:
    """Names, #hashtags and @mentions. Sentence-initial capitals are ignored."""
    out = {t.lower() for t in _TAGISH.findall(text)}
    for m in _CAP.finditer(text):
        before = text[: m.start()].rstrip()
        if before and before[-1] not in ".!?:\n":
            w = m.group().lower()
            if w not in STOP:
                out.add(w)
    return out


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 3]


def tokens_est(text: str) -> int:
    """Rough token count (~4 chars/token). Good enough for budgeting."""
    return math.ceil(len(text) / 4) + 2


def guess_importance(text: str) -> float:
    low = text.lower()
    imp = 0.4
    if any(c in low for c in CUES):
        imp += 0.3
    if re.search(r"\d", text):
        imp += 0.1
    if "!" in text:
        imp += 0.05
    return min(imp, 0.95)


def summarize(texts: Iterable[str], max_chars: int = 300) -> str:
    """Zero-dependency extractive summary: keep the sentences that carry the cluster's themes.

    Swap this for an LLM by passing ``summarizer=`` to :class:`sediment.Sediment`.
    """
    sents = [s for t in texts for s in split_sentences(t)]
    if not sents:
        return ""
    df: Counter = Counter()
    for s in sents:
        df.update(set(keywords(s)))
    scored = []
    for i, s in enumerate(sents):
        ks = keywords(s)
        if ks:
            scored.append((sum(df[k] for k in set(ks)) / math.sqrt(len(ks)), i, s))
    if not scored:
        return sents[0][:max_chars]
    chosen, seen, used = [], [], 0
    for _, i, s in sorted(scored, key=lambda x: (-x[0], x[1])):
        sig = frozenset(keywords(s))
        if any(len(sig & o) / max(1, len(sig | o)) > 0.5 for o in seen):
            continue
        if len(s) > max_chars:
            s = s[: max_chars - 1].rstrip() + "…"
        if used + len(s) + 1 > max_chars and chosen:
            continue
        chosen.append((i, s))
        seen.append(sig)
        used += len(s) + 1
    parts = []
    for _, s in sorted(chosen):
        parts.append(s if s[-1] in ".!?…" else s + ".")
    return " ".join(parts)


def ago(seconds: float) -> str:
    d = max(0.0, seconds) / 86400.0
    if d < 1:
        return "today"
    if d < 14:
        return f"{int(d)}d ago"
    if d < 60:
        return f"{int(d // 7)}w ago"
    if d < 730:
        return f"{int(d // 30)}mo ago"
    return f"{int(d // 365)}y ago"
