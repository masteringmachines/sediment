import os
import tempfile
import unittest

from sediment import Sediment
from sediment.core import DAY


class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t

    def advance(self, days):
        self.t += days * DAY


class SedimentTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = Sediment(":memory:", clock=self.clock)
        self.addCleanup(self.m.close)

    def test_remember_and_recall(self):
        self.m.remember("Maya benchmarked pgvector against Qdrant for Atlas.")
        self.m.remember("The cafeteria serves tacos on Tuesdays.")
        r = self.m.recall("why pgvector?")
        self.assertEqual(len(r.hits), 1)
        self.assertIn("pgvector", r.hits[0].text)
        self.assertIn("pgvector", r.render())

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            self.m.remember("   ")

    def test_empty_recall_is_falsy(self):
        self.assertFalse(self.m.recall("nothing here"))
        self.assertEqual(self.m.context("nothing here"), "")

    def test_budget_is_respected(self):
        for i in range(30):
            self.m.remember(f"Deploy note {i}: the staging cluster needs a restart after release {i}.")
        r = self.m.recall("staging cluster restart", budget=60, k=30)
        self.assertLessEqual(r.tokens, 60 + 25)  # one oversized first hit may exceed; others cannot
        self.assertLess(len(r.hits), 30)

    def test_forgetting_curve_and_reinforcement(self):
        mid = self.m.remember("Temporary wifi password is hunter2 for the guest network.", importance=0.5)
        s0 = self.m.get(mid)["stability"]
        self.clock.advance(6)
        self.m.recall("guest wifi password")
        s1 = self.m.get(mid)["stability"]
        self.assertGreater(s1, s0)  # recall strengthens memory

    def test_sleep_compacts_and_nothing_is_lost(self):
        ids = [self.m.remember(f"Atlas standup {i}: Maya fixed the billing bug number {i}.") for i in range(5)]
        self.m.remember("Decision: Atlas uses PostgreSQL with pgvector.")
        self.clock.advance(120)
        rep = self.m.sleep()
        self.assertGreaterEqual(rep.fossilized, 6)
        self.assertLess(rep.hot_after, rep.hot_before)
        stats = self.m.stats()
        self.assertEqual(stats["layers"][0], 0)
        self.assertEqual(stats["fossils"], 6)
        # every original is recoverable from some gist
        gists = [r["id"] for r in self.m.db.execute("SELECT id FROM memories WHERE layer>0")]
        recovered = {leaf.id for g in gists for leaf in self.m.dig(g)}
        self.assertTrue(set(ids).issubset(recovered))

    def test_deep_recall_finds_detail_summary_dropped(self):
        texts = [f"Nimbus sprint {i}: shipped the queue refactor and updated the dashboards." for i in range(6)]
        texts.append("Nimbus incident: Redis eviction policy caused the outage.")
        for t in texts:
            self.m.remember(t)
        self.clock.advance(120)
        self.m.sleep(summarizer=lambda ts, n: "Nimbus had a busy stretch.")  # lossy on purpose
        r = self.m.recall("what caused the Redis outage in Nimbus?")
        self.assertTrue(r.hits)
        self.assertIn("Redis", r.render())  # scent found the gist, fossils supplied the detail

    def test_pinned_never_fades(self):
        pid = self.m.remember("The user's name is Sam.", pin=True)
        self.clock.advance(3650)
        self.m.sleep()
        self.assertEqual(self.m.get(pid)["layer"], 0)
        self.assertIn("Sam", self.m.recall("user name").render())

    def test_gists_merge_into_bedrock(self):
        for wave in range(3):
            for i in range(3):
                self.m.remember(f"Harbor release {wave}-{i}: Leo updated the deploy pipeline step {i}.")
            self.clock.advance(40)
            self.m.sleep()
        self.clock.advance(2000)
        self.m.sleep()
        self.assertGreaterEqual(self.m.stats()["layers"][2], 1)
        self.assertEqual(self.m.stats()["fossils"] > 0, True)

    def test_association_pulls_related_memory(self):
        self.m.remember("Our Nimbus billing service is owned by Dana.")
        self.m.remember("The invoice run crashed inside Nimbus billing.")
        r = self.m.recall("invoice run crash")
        self.assertIn("Dana", r.render())
        self.assertEqual([h.via for h in r.hits], ["match", "assoc"])

    def test_erase_removes_everything_beneath(self):
        for i in range(4):
            self.m.remember(f"Secret project Zephyr detail {i}: launch codes are shared.")
        self.clock.advance(200)
        self.m.sleep()
        gid = self.m.db.execute("SELECT id FROM memories WHERE layer>0").fetchone()["id"]
        self.assertGreater(self.m.erase(gid), 0)
        self.assertEqual(self.m.stats()["fossils"], 0)
        self.assertFalse(self.m.recall("Zephyr launch codes"))

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.db")
            with Sediment(path, clock=self.clock) as a:
                a.remember("Ravi's birthday is March 14.")
            with Sediment(path, clock=self.clock) as b:
                self.assertIn("March 14", b.recall("Ravi birthday").render())

    def test_ids_never_collide_after_compaction(self):
        for i in range(4):
            self.m.remember(f"Collision test {i} about the Orion cache layer.")
        self.clock.advance(300)
        self.m.sleep()
        new = self.m.remember("Fresh memory after compaction.")
        fossil_ids = {r[0] for r in self.m.db.execute("SELECT id FROM fossils")}
        self.assertNotIn(new, fossil_ids)


if __name__ == "__main__":
    unittest.main()
