"""Sediment: memory that settles instead of overflowing.

Layers (hot, searchable)      Cold storage
  0 fresh   full episodes       fossils: every original, zlib-compressed,
  1 gist    compacted summaries          never deleted, dug up on demand
  2 bedrock long-term themes
"""
from __future__ import annotations

import math
import sqlite3
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from . import text as T

DAY = 86400.0
Summarizer = Callable[[List[str], int], str]
LAYER_NAMES = {0: "fresh", 1: "gist", 2: "bedrock"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, layer INTEGER NOT NULL DEFAULT 0, text TEXT NOT NULL,
  importance REAL NOT NULL, stability REAL NOT NULL, last_access REAL NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
  tags TEXT NOT NULL DEFAULT '', scent TEXT NOT NULL DEFAULT '',
  span_start REAL, span_end REAL, source TEXT);
CREATE TABLE IF NOT EXISTS links(ent TEXT NOT NULL, mem INTEGER NOT NULL, PRIMARY KEY(ent, mem));
CREATE INDEX IF NOT EXISTS links_mem ON links(mem);
CREATE TABLE IF NOT EXISTS fossils(
  id INTEGER PRIMARY KEY, parent INTEGER NOT NULL, layer INTEGER NOT NULL,
  ts REAL NOT NULL, size INTEGER NOT NULL, blob BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS fossils_parent ON fossils(parent);
"""


class Leaf(NamedTuple):
    id: int
    ts: float
    text: str


@dataclass
class Hit:
    id: int
    layer: int
    text: str
    score: float
    retention: float
    ts: float
    via: str = "match"  # "match" | "assoc"
    detail: str = ""  # sentence dug out of the fossils beneath a gist
    span: Optional[Tuple[float, float]] = None


@dataclass
class Recollection:
    query: str
    hits: List[Hit]
    tokens: int
    now: float

    def __bool__(self) -> bool:
        return bool(self.hits)

    def render(self, header: bool = True) -> str:
        """Prompt-ready text block."""
        if not self.hits:
            return ""
        lines = []
        if header:
            lines.append(f"[sediment: {len(self.hits)} memories, ~{self.tokens} tokens]")
        for h in self.hits:
            end = h.span[1] if h.span else h.ts
            label = ("" if h.layer == 0 else LAYER_NAMES[h.layer] + ", ") + T.ago(self.now - end)
            line = f"- ({label}) {h.text}"
            if h.detail:
                line += f" ↳ {h.detail}"
            lines.append(line)
        return "\n".join(lines)

    __str__ = render


@dataclass
class SleepReport:
    faded: int = 0
    gists: int = 0
    fossilized: int = 0
    hot_before: int = 0
    hot_after: int = 0


class Sediment:
    """Unbounded memory with bounded recall cost.

    >>> m = Sediment(":memory:")
    >>> m.remember("Maya benchmarked pgvector against Qdrant for Atlas.")  # doctest: +SKIP
    >>> print(m.recall("why pgvector?"))                                   # doctest: +SKIP
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        clock: Callable[[], float] = time.time,
        summarizer: Optional[Summarizer] = None,
        base_days: float = 7.0,
    ):
        self.path = str(path)
        self.clock = clock
        self.summarizer: Summarizer = summarizer or T.summarize
        self.base_days = base_days
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.fts = self._init_fts()

    # ---------------------------------------------------------------- plumbing
    def _init_fts(self) -> bool:
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
                "text, tags, scent, tokenize='porter unicode61')"
            )
            return True
        except sqlite3.OperationalError:  # SQLite built without FTS5: slower fallback
            return False

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> "Sediment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _tagstr(tags: Iterable[str]) -> str:
        return ",".join(sorted({t.strip().lower() for t in tags if t.strip()}))

    def _index(self, mid: int, text: str, tags: str, scent: str, extra_ents: Iterable[str] = ()) -> None:
        ents = T.entities(text) | {t for t in tags.split(",") if t} | set(extra_ents)
        self.db.executemany("INSERT OR IGNORE INTO links VALUES(?,?)", [(e, mid) for e in ents])
        if self.fts:
            self.db.execute(
                "INSERT INTO fts(rowid,text,tags,scent) VALUES(?,?,?,?)",
                (mid, text, tags.replace(",", " "), scent),
            )

    def _unindex(self, mid: int) -> None:
        self.db.execute("DELETE FROM links WHERE mem=?", (mid,))
        if self.fts:
            self.db.execute("DELETE FROM fts WHERE rowid=?", (mid,))

    # ------------------------------------------------------------------ memory
    @staticmethod
    def _retention(row: sqlite3.Row, now: float) -> float:
        if row["pinned"]:
            return 1.0
        dt = max(0.0, now - row["last_access"]) / DAY
        return math.exp(-dt / max(row["stability"], 1e-6))

    def remember(
        self,
        text: str,
        *,
        tags: Iterable[str] = (),
        importance: Optional[float] = None,
        pin: bool = False,
        source: Optional[str] = None,
    ) -> int:
        """Store a memory. ``pin=True`` makes it immune to fading. Returns its id."""
        text = text.strip()
        if not text:
            raise ValueError("cannot remember empty text")
        now = self.clock()
        imp = 1.0 if pin else min(1.0, max(0.0, importance if importance is not None else T.guess_importance(text)))
        tags_s = self._tagstr(tags)
        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories(ts,layer,text,importance,stability,last_access,pinned,tags,source)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (now, 0, text, imp, self.base_days * (0.5 + imp), now, int(pin), tags_s, source),
            )
            mid = int(cur.lastrowid)
            self._index(mid, text, tags_s, "")
        return mid

    def get(self, mid: int) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()

    # ------------------------------------------------------------------ recall
    def _search(self, qk: Sequence[str]) -> List[Tuple[int, float]]:
        if not qk:
            return []
        if self.fts:
            q = " OR ".join(f'"{w}"' for w in dict.fromkeys(qk))
            try:
                rows = self.db.execute(
                    "SELECT rowid, bm25(fts, 1.0, 1.5, 0.5) AS r FROM fts WHERE fts MATCH ? ORDER BY r LIMIT 60",
                    (q,),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            if not rows:
                return []
            raw = [max(0.0, -r["r"]) for r in rows]
            top = max(raw) or 1.0
            return [(r["rowid"], 0.3 + 0.7 * x / top) for r, x in zip(rows, raw)]
        qs = {T.stem(w) for w in qk}
        out = []
        for r in self.db.execute("SELECT id, text, tags, scent FROM memories"):
            ov = len(qs & T.kwset(f"{r['text']} {r['tags']} {r['scent']}"))
            if ov:
                out.append((r["id"], 0.3 + 0.7 * ov / len(qs)))
        return sorted(out, key=lambda x: -x[1])[:60]

    def _assoc(self, cand: Dict[int, List[float]], query: str, qk: Sequence[str]) -> None:
        """Spreading activation: memories sharing entities with the hits (or the query) get a nudge."""
        seeds = sorted(cand, key=lambda m: -cand[m][0])[:3]
        ents = T.entities(query) | set(qk)
        if seeds:
            ph = ",".join("?" * len(seeds))
            ents |= {r[0] for r in self.db.execute(f"SELECT ent FROM links WHERE mem IN ({ph})", seeds)}
        ents = list(ents)[:200]
        if not ents:
            return
        ph = ",".join("?" * len(ents))
        # Rare entities (a person, a one-off project) link memories; ubiquitous ones do not.
        df = {r[0]: r[1] for r in self.db.execute(
            f"SELECT ent, COUNT(*) FROM links WHERE ent IN ({ph}) GROUP BY ent", ents)}
        weight = {e: (1.0 if n <= 3 else 0.5 if n <= 8 else 0.1 if n <= 50 else 0.0) for e, n in df.items()}
        score: Dict[int, float] = {}
        for ent, mem in self.db.execute(f"SELECT ent, mem FROM links WHERE ent IN ({ph})", ents):
            if weight.get(ent):
                score[mem] = score.get(mem, 0.0) + weight[ent]
        for mem, w in sorted(score.items(), key=lambda x: -x[1])[:40]:
            cand.setdefault(mem, [0.0, 0.0])[1] = min(1.0, w / 1.5)

    def _leaves(self, mid: int, limit: int = 500) -> List[Leaf]:
        out: List[Leaf] = []
        stack = [mid]
        while stack and len(out) < limit:
            p = stack.pop()
            for r in self.db.execute("SELECT id, layer, ts, blob FROM fossils WHERE parent=?", (p,)):
                if r["layer"] == 0:
                    out.append(Leaf(r["id"], r["ts"], zlib.decompress(r["blob"]).decode()))
                else:
                    stack.append(r["id"])
        return sorted(out, key=lambda leaf: leaf.ts)

    def dig(self, mid: int) -> List[Leaf]:
        """Excavate the original, unabridged memories that a gist/bedrock row was built from."""
        row = self.get(mid)
        if row is None:
            return []
        if row["layer"] == 0:
            return [Leaf(mid, row["ts"], row["text"])]
        return self._leaves(mid, limit=10**9)

    def _detail(self, mid: int, qs: set, gist: str) -> str:
        best, best_s = "", min(2, len(qs)) - 1  # need >=2 overlapping terms (1 for 1-word queries)
        for leaf in self._leaves(mid):
            for s in T.split_sentences(leaf.text):
                ov = len(qs & T.kwset(s))
                if ov > best_s:
                    best, best_s = s, ov
        if not best or best in gist:
            return ""
        return best if len(best) <= 240 else best[:239] + "…"

    def recall(
        self,
        query: str,
        *,
        budget: int = 600,
        k: int = 8,
        reinforce: bool = True,
        deep: bool = True,
        min_ratio: float = 0.75,
        associate: int = 1,
    ) -> Recollection:
        """Return the memories most worth spending ``budget`` tokens on.

        Score = relevance (BM25) + how well it is still remembered + association.
        Hits scoring below ``min_ratio`` x the best hit are dropped as noise, except up to
        ``associate`` memories that merely share people/projects/tags with what was found.
        Recalling a memory strengthens it (spaced-repetition effect).
        """
        now = self.clock()
        qk = T.keywords(query)
        cand: Dict[int, List[float]] = {m: [rel, 0.0] for m, rel in self._search(qk)}
        self._assoc(cand, query, qk)
        if not cand:
            return Recollection(query, [], 0, now)
        ph = ",".join("?" * len(cand))
        rows = {r["id"]: r for r in self.db.execute(f"SELECT * FROM memories WHERE id IN ({ph})", list(cand))}
        ranked: List[Hit] = []
        for mid, (rel, assoc) in cand.items():
            r = rows.get(mid)
            if r is None:
                continue
            ret = self._retention(r, now)
            strength = ret * (0.4 + 0.6 * r["importance"])
            score = 0.62 * rel + 0.23 * strength + 0.15 * assoc
            span = (r["span_start"], r["span_end"]) if r["span_start"] is not None else None
            ranked.append(Hit(mid, r["layer"], r["text"], score, ret, r["ts"], "match" if rel else "assoc", span=span))
        ranked.sort(key=lambda h: -h.score)

        qs = {T.stem(w) for w in qk}
        hits: List[Hit] = []
        used = 0
        floor = ranked[0].score * min_ratio
        assoc_left = associate
        for h in ranked:
            if len(hits) >= k:
                break
            if h.via == "match" and h.score < floor:
                continue
            if h.via == "assoc":
                if assoc_left <= 0 or cand[h.id][1] < 0.6:  # needs a rare shared entity
                    continue
                assoc_left -= 1
            if deep and h.layer > 0 and qs:
                h.detail = self._detail(h.id, qs, h.text)
            cost = T.tokens_est(h.text + h.detail)
            if used + cost > budget and hits:
                continue
            hits.append(h)
            used += cost
        if reinforce and hits:
            with self.db:
                for h in hits:
                    r = rows[h.id]
                    stab = min(3650.0, r["stability"] * (1.3 + 1.2 * (1 - h.retention)))
                    self.db.execute(
                        "UPDATE memories SET stability=?, last_access=?, hits=hits+1 WHERE id=?",
                        (stab, now, h.id),
                    )
        return Recollection(query, hits, used, now)

    def context(self, query: str, budget: int = 600) -> str:
        """Shorthand: a prompt-ready block of memories for ``query`` (empty string if none)."""
        return self.recall(query, budget=budget).render()

    # ------------------------------------------------------------------- sleep
    def _entities_of(self, mid: int) -> set:
        return {r[0] for r in self.db.execute("SELECT ent FROM links WHERE mem=?", (mid,))}

    def _cluster(self, rows: List[sqlite3.Row]) -> List[List[sqlite3.Row]]:
        clusters: List[dict] = []
        for r in sorted(rows, key=lambda r: r["ts"]):
            ks, es = T.kwset(r["text"]), self._entities_of(r["id"])
            best, best_s = None, 0.0
            for c in clusters:
                j = max((len(ks & o) / len(ks | o) for o in c["kws"]), default=0.0) if ks else 0.0
                s = j + 0.12 * min(len(es & c["ents"]), 3)
                if s > best_s:
                    best, best_s = c, s
            if best is not None and best_s >= 0.3:
                best["rows"].append(r)
                best["kws"].append(ks)
                best["ents"] |= es
            else:
                clusters.append({"rows": [r], "kws": [ks], "ents": set(es)})
        return [c["rows"] for c in clusters]

    def _hot_bytes(self) -> int:
        return int(self.db.execute("SELECT COALESCE(SUM(LENGTH(text)),0) FROM memories").fetchone()[0])

    def sleep(self, *, threshold: float = 0.25, summarizer: Optional[Summarizer] = None) -> SleepReport:
        """Consolidate: memories that have faded below ``threshold`` are compacted a layer down.

        Originals move to compressed fossils (still retrievable). A gist keeps a
        'scent' of every keyword it absorbed, so details the summary dropped are
        still findable. Entity links survive too.
        """
        summ = summarizer or self.summarizer
        now = self.clock()
        rep = SleepReport(hot_before=self._hot_bytes())
        for layer in (0, 1):
            rows = [
                r
                for r in self.db.execute("SELECT * FROM memories WHERE layer=? AND pinned=0", (layer,))
                if self._retention(r, now) < threshold
            ]
            rep.faded += len(rows)
            for group in self._cluster(rows):
                if layer == 1 and len(group) < 2:
                    continue  # lone gists stay as they are
                self._compact(group, layer, summ, now, rep)
        rep.hot_after = self._hot_bytes()
        return rep

    def _compact(self, group: List[sqlite3.Row], layer: int, summ: Summarizer, now: float, rep: SleepReport) -> None:
        max_chars = min(360, 180 + 50 * len(group))
        texts = [r["text"] for r in group]
        gist = (summ(texts, max_chars) or "").strip() or T.summarize(texts, max_chars) or texts[0][:max_chars]
        scent: Counter = Counter()
        for r in group:
            scent.update(T.keywords(r["text"]))
            scent.update(r["scent"].split())
        scent_s = " ".join(w for w, _ in sorted(scent.items(), key=lambda x: (-x[1], x[0]))[:80])
        tags = self._tagstr(t for r in group for t in r["tags"].split(","))
        tags = ",".join(tags.split(",")[:8]) if tags else ""
        ents = set().union(*(self._entities_of(r["id"]) for r in group))
        imp = max(r["importance"] for r in group)
        starts = [r["span_start"] if r["span_start"] is not None else r["ts"] for r in group]
        ends = [r["span_end"] if r["span_end"] is not None else r["ts"] for r in group]
        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories(ts,layer,text,importance,stability,last_access,hits,tags,scent,span_start,span_end)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (max(r["ts"] for r in group), layer + 1, gist, imp,
                 self.base_days * (0.5 + imp) * 4 ** (layer + 1), now,
                 sum(r["hits"] for r in group), tags, scent_s, min(starts), max(ends)),
            )
            gid = int(cur.lastrowid)
            for r in group:
                blob = zlib.compress(r["text"].encode(), 9)
                self.db.execute(
                    "INSERT INTO fossils(id,parent,layer,ts,size,blob) VALUES(?,?,?,?,?,?)",
                    (r["id"], gid, r["layer"], r["ts"], len(r["text"].encode()), blob),
                )
                self._unindex(r["id"])
                self.db.execute("DELETE FROM memories WHERE id=?", (r["id"],))
            self._index(gid, gist, tags, scent_s, ents)
        rep.gists += 1
        rep.fossilized += len(group)

    # ------------------------------------------------------------ housekeeping
    def erase(self, mid: int) -> int:
        """Hard-delete a memory *and* every fossil beneath it (privacy / right-to-be-forgotten)."""
        removed, stack = 0, [mid]
        with self.db:
            while stack:
                p = stack.pop()
                for r in self.db.execute("SELECT id, layer FROM fossils WHERE parent=?", (p,)).fetchall():
                    if r["layer"] > 0:
                        stack.append(r["id"])
                    removed += 1
                self.db.execute("DELETE FROM fossils WHERE parent=?", (p,))
            cur = self.db.execute("SELECT 1 FROM memories WHERE id=?", (mid,)).fetchone()
            if cur:
                self._unindex(mid)
                self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
                removed += 1
            else:
                removed += self.db.execute("DELETE FROM fossils WHERE id=?", (mid,)).rowcount
        return removed

    def stats(self) -> dict:
        layers = {0: 0, 1: 0, 2: 0}
        for r in self.db.execute("SELECT layer, COUNT(*) c FROM memories GROUP BY layer"):
            layers[r["layer"]] = r["c"]
        f = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(LENGTH(blob)),0) z,"
            " COALESCE(SUM(CASE WHEN layer=0 THEN size END),0) raw FROM fossils"
        ).fetchone()
        hot = self._hot_bytes()
        fresh = int(
            self.db.execute("SELECT COALESCE(SUM(LENGTH(text)),0) FROM memories WHERE layer=0").fetchone()[0]
        )
        raw = fresh + int(f["raw"])
        return {
            "layers": layers,
            "fossils": int(f["n"]),
            "hot_bytes": hot,
            "fossil_bytes": int(f["z"]),
            "raw_bytes": raw,
            "compression": (raw / hot) if hot else 1.0,
        }
