import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from tokenaudit.loader import load_sessions
from tokenaudit.model import Session, Turn, Usage
from tokenaudit.pricing import Pricing
from tokenaudit.report import (NOTE_CACHE_WRITE_12H, NOTE_CACHE_WRITE_30M, build_report,
                               render_markdown)
from tokenaudit.rules import detect
from tests.fixtures.builder import Rollout, meta_line, sessions_dir, turn_context_line, write_rollout

PRICING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills", "token-audit", "scripts", "pricing.json")
S1 = "019a0000-0000-7000-8000-000000000001"
S1A = "019a0000-0000-7000-8000-00000000001a"
S2 = "019a0000-0000-7000-8000-000000000002"
TS = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _session(usage, model="gpt-5.5", sid="s1", subagents=None):
    turn = Turn(response_id="r0", ts=TS, model=model, usage=usage)
    return Session(session_id=sid, project="/p", path="x", turns=[turn],
                   subagents=list(subagents or []))


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pricing = Pricing.load(PRICING_PATH)
        r = Rollout()
        write_rollout(self.tmp.name, S1, [
            meta_line(S1, "2026-09-01T10:00:00Z", cwd="/w/alpha|work"),
            turn_context_line("2026-09-01T10:00:00Z", model="gpt-5.5", effort="xhigh"),
            r.token_count_line("2026-09-01T10:00:00Z", {"input": 30100, "cached": 0, "output": 1000, "reasoning": 400}),
            r.token_count_line("2026-09-01T13:00:00Z", {"input": 77503, "cached": 100, "output": 500, "reasoning": 100}),
        ], ts="2026-09-01T10:00:00Z")
        write_rollout(self.tmp.name, S1A, [
            meta_line(S1A, "2026-09-01T10:00:30Z", cwd="/w/alpha|work", parent_id=S1),
            turn_context_line("2026-09-01T10:00:30Z", model="gpt-5.6-terra"),
            Rollout().token_count_line("2026-09-01T10:00:30Z", {"input": 1000, "output": 200}),
        ], ts="2026-09-01T10:00:30Z")
        write_rollout(self.tmp.name, S2, [
            meta_line(S2, "2026-09-02T10:00:00Z", cwd="/w/beta"),
            turn_context_line("2026-09-02T10:00:00Z", model="gpt-weird-1"),
            Rollout().token_count_line("2026-09-02T10:00:00Z", {"input": 10, "output": 10}),
        ], ts="2026-09-02T10:00:00Z")
        self.sessions = load_sessions(sessions_dir(self.tmp.name), all_time=True)
        self.findings = sorted(
            (f for s in self.sessions for f in detect(s, self.pricing)),
            key=lambda f: -f.waste_usd)
        self.report = build_report(self.sessions, self.findings, self.pricing, top=10, days=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_totals_and_models(self):
        r = self.report
        self.assertEqual(r["sessions"], 2)
        self.assertEqual(r["period"]["days"], None)
        self.assertEqual(r["period"]["from"][:10], "2026-09-01")
        self.assertEqual(r["period"]["to"][:10], "2026-09-02")
        expected_total = (
            30100 * 5e-6 + 1000 * 30e-6
            + (77503 - 100) * 5e-6 + 100 * 0.1 * 5e-6 + 500 * 30e-6
            + 1000 * 2e-6 + 200 * 12e-6
            + 10 * 5e-6 + 10 * 30e-6)
        self.assertAlmostEqual(r["totals"]["total"], expected_total, places=6)
        models = [m["model"] for m in r["by_model"]]
        self.assertEqual(models[0], "gpt-5.5")
        self.assertIn("gpt-5.6-terra", models)
        self.assertAlmostEqual(sum(m["share"] for m in r["by_model"]), 1.0, places=6)

    def test_findings_and_rule_summary(self):
        r = self.report
        self.assertEqual(r["findings"][0]["rule"], "R1")
        self.assertEqual(r["findings"][0]["session_name"], "alpha|work 019a0000")
        rules = {b["rule"]: b for b in r["by_rule"]}
        self.assertEqual(rules["R1"]["count"], 1)

    def test_info_section(self):
        info = self.report["info"]
        self.assertEqual(info["subagents"]["count"], 1)
        self.assertAlmostEqual(info["subagents"]["usd"], 1000 * 2e-6 + 200 * 12e-6)
        self.assertAlmostEqual(info["subagents"]["share"], info["subagents"]["usd"] / self.report["totals"]["total"])
        self.assertEqual(info["top_output_sessions"][0]["name"], "alpha|work 019a0000")
        self.assertEqual(info["reasoning"], {"tokens": 500, "output_tokens": 1710, "share": 500 / 1710})
        self.assertEqual(info["effort_sessions"], {"xhigh": 1})
        self.assertIn("gpt-5.5", info["expensive_models"])
        self.assertEqual(info["short_sessions_on_expensive_models"], 2)  # s1 (5.5) and s2 (unknown->5.5)
        self.assertEqual(info["unknown_models"], ["gpt-weird-1"])
        self.assertEqual(info["skipped_files"], 0)
        self.assertIsNone(info["cache_write_note"])

    def test_json_serializable_and_top_limit(self):
        json.dumps(self.report)
        small = build_report(self.sessions, self.findings, self.pricing, top=0, days=7, skipped_files=2)
        self.assertEqual(small["findings"], [])
        self.assertEqual(small["period"]["days"], 7)
        self.assertIn("Skipped 2 compressed", render_markdown(small))

    def test_markdown_sections_and_pipe_escaping(self):
        md = render_markdown(self.report)
        for heading in ("# codex token-audit report", "## Cost by category", "## Cost by model",
                        "## Top findings", "## Findings by rule", "## Info"):
            self.assertIn(heading, md)
        self.assertIn("alpha\\|work", md)
        self.assertIn("| R1 |", md)
        self.assertIn("gpt-weird-1", md)
        self.assertIn("Reasoning: 500 of 1710", md)
        self.assertIn("xhigh=1", md)
        self.assertIn("% of total)", md)
        self.assertLess(len(md.encode("utf-8")), 8192)

    def test_empty_input(self):
        r = build_report([], [], self.pricing, top=10, days=30)
        self.assertEqual(r["sessions"], 0)
        self.assertEqual(r["totals"]["total"], 0.0)
        self.assertEqual(r["info"]["subagents"]["share"], 0.0)
        self.assertEqual(r["info"]["reasoning"]["share"], 0.0)
        md = render_markdown(r)
        self.assertIn("Sessions: 0", md)
        self.assertIsNone(r["info"]["cache_write_note"])
        self.assertNotIn("1.25x input", md)


class CacheWriteNoteTest(unittest.TestCase):
    def setUp(self):
        self.pricing = Pricing.load(PRICING_PATH)

    def test_assumed_30m_when_writes_have_no_ttl_split(self):
        r = build_report([_session(Usage(input=1000, cache_write=400))], [], self.pricing)
        self.assertEqual(r["info"]["cache_write_note"], NOTE_CACHE_WRITE_30M)
        md = render_markdown(r)
        self.assertIn(NOTE_CACHE_WRITE_30M, md)
        self.assertIn("up to 2x the amount shown", md)

    def test_12h_config_note_when_ttl_override_set(self):
        p = Pricing({
            "default_model": "gpt-5.5",
            "cache_write_ttl": "12h",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {"gpt-5.5": {"input": 5, "output": 30}},
        })
        r = build_report([_session(Usage(input=1000, cache_write=400))], [], p)
        self.assertEqual(r["info"]["cache_write_note"], NOTE_CACHE_WRITE_12H)
        md = render_markdown(r)
        self.assertIn(NOTE_CACHE_WRITE_12H, md)
        self.assertNotIn(NOTE_CACHE_WRITE_30M, md)

    def test_per_model_12h_ttl_uses_config_note(self):
        p = Pricing({
            "default_model": "a",
            "cache_write_ttl": "30m",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {
                "a": {"input": 5, "output": 30},
                "b": {"input": 5, "output": 30, "cache_write_ttl": "12h"},
            },
        })
        r = build_report([_session(Usage(input=1000, cache_write=400), model="b")], [], p)
        self.assertEqual(r["info"]["cache_write_note"], NOTE_CACHE_WRITE_12H)

    def test_no_note_when_log_split_present(self):
        r = build_report(
            [_session(Usage(input=1000, cache_write=400, cache_write_12h=100))],
            [], self.pricing)
        self.assertIsNone(r["info"]["cache_write_note"])
        self.assertNotIn("1.25x input", render_markdown(r))

    def test_no_note_when_split_is_explicit_zero(self):
        r = build_report(
            [_session(Usage(input=1000, cache_write=400, cache_write_12h=0))],
            [], self.pricing)
        self.assertIsNone(r["info"]["cache_write_note"])

    def test_subagent_writes_trigger_note(self):
        child = _session(Usage(input=200, cache_write=50), sid="child")
        child.source = "subagent:thread_spawn"
        parent = _session(Usage(input=10), sid="parent", subagents=[child])
        r = build_report([parent], [], self.pricing)
        self.assertEqual(r["info"]["cache_write_note"], NOTE_CACHE_WRITE_30M)

    def test_mixed_30m_and_12h_config_prefers_30m_caveat(self):
        p = Pricing({
            "default_model": "a",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {
                "a": {"input": 5, "output": 30},
                "b": {"input": 5, "output": 30, "cache_write_ttl": "12h"},
            },
        })
        sessions = [
            _session(Usage(input=100, cache_write=10), model="a", sid="s-a"),
            _session(Usage(input=100, cache_write=10), model="b", sid="s-b"),
        ]
        r = build_report(sessions, [], p)
        self.assertEqual(r["info"]["cache_write_note"], NOTE_CACHE_WRITE_30M)


if __name__ == "__main__":
    unittest.main()
