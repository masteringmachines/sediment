"""Command line: ``sediment remember | recall | sleep | dig | strata | demo``."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .core import Sediment


def _bar(n: int, top: int, width: int = 24, ch: str = "█") -> str:
    fill = 0 if n == 0 else max(1, round(width * n / max(top, 1)))
    return ch * fill + "·" * (width - fill)


def _kb(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def render_strata(m: Sediment) -> str:
    s = m.stats()
    L = s["layers"]
    top = max(L[0], L[1], L[2], s["fossils"], 1)
    rows = [
        ("░", "fresh", L[0], "recent episodes, full detail"),
        ("▒", "gist", L[1], "faded → compact summaries"),
        ("▓", "bedrock", L[2], "long-term themes"),
        ("█", "fossils", s["fossils"], "raw originals, zlib, never deleted"),
    ]
    out = ["", "  SEDIMENT · core sample", "  " + "─" * 66]
    for glyph, name, n, note in rows:
        out.append(f"  {glyph} {name:<8}{_bar(n, top)} {n:>5}  {note}")
    out.append("  " + "─" * 66)
    out.append(
        f"  hot index {_kb(s['hot_bytes'])} · raw history {_kb(s['raw_bytes'])} · "
        f"cold storage {_kb(s['fossil_bytes'])} · {s['compression']:.1f}× compression"
    )
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="sediment", description="Infinite memory for AI agents.")
    p.add_argument("--db", default=os.environ.get("SEDIMENT_DB", "sediment.db"), help="database file")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("remember", help="store a memory")
    a.add_argument("text", nargs="+")
    a.add_argument("-t", "--tag", action="append", default=[])
    a.add_argument("-i", "--importance", type=float)
    a.add_argument("--pin", action="store_true", help="never let this fade")

    a = sub.add_parser("recall", help="search memory")
    a.add_argument("query", nargs="+")
    a.add_argument("-b", "--budget", type=int, default=600, help="token budget")

    a = sub.add_parser("sleep", help="compact faded memories")
    a.add_argument("--threshold", type=float, default=0.25)

    a = sub.add_parser("dig", help="excavate the originals beneath a gist")
    a.add_argument("id", type=int)

    sub.add_parser("strata", help="show the core sample")
    a = sub.add_parser("demo", help="150-day time-lapse")
    a.add_argument("--days", type=int, default=150)

    args = p.parse_args(argv)
    if args.cmd == "demo":
        from .demo import run

        run(args.days)
        return 0

    with Sediment(args.db) as m:
        if args.cmd == "remember":
            mid = m.remember(" ".join(args.text), tags=args.tag, importance=args.importance, pin=args.pin)
            print(f"remembered #{mid}")
        elif args.cmd == "recall":
            r = m.recall(" ".join(args.query), budget=args.budget)
            print(r.render() or "(nothing recalled)")
        elif args.cmd == "sleep":
            r = m.sleep(threshold=args.threshold)
            print(f"{r.faded} faded → {r.gists} gists, {r.fossilized} fossilized · hot {r.hot_before}B → {r.hot_after}B")
        elif args.cmd == "dig":
            leaves = m.dig(args.id)
            if not leaves:
                print(f"nothing under #{args.id}", file=sys.stderr)
                return 1
            for leaf in leaves:
                print(f"#{leaf.id}: {leaf.text}")
        elif args.cmd == "strata":
            print(render_strata(m))
    return 0
