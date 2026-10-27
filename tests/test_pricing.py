import os
import unittest

from tokenaudit.model import Usage
from tokenaudit.pricing import Cost, Pricing

PRICING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills", "token-audit", "scripts", "pricing.json",
)


class PricingTest(unittest.TestCase):
    def setUp(self):
        self.p = Pricing.load(PRICING_PATH)

    def test_gpt55_cost_breakdown(self):
        u = Usage(input=12000, cached=10000, cache_write=1000, output=100, reasoning=40)
        c = self.p.cost("gpt-5.5", u)
        self.assertAlmostEqual(c.input, 1000 * 5e-6)
        self.assertAlmostEqual(c.cache_write, 1000 * 1.25 * 5e-6)
        self.assertAlmostEqual(c.cache_read, 10000 * 0.1 * 5e-6)
        self.assertAlmostEqual(c.output, 100 * 30e-6)
        self.assertAlmostEqual(c.total, c.input + c.cache_write + c.cache_read + c.output)

    def test_per_model_cache_read_override(self):
        c = self.p.cost("o3", Usage(input=10000, cached=10000))
        self.assertAlmostEqual(c.cache_read, 10000 * 0.25 * 2e-6)
        self.assertAlmostEqual(c.input, 0.0)

    def test_prefix_match_requires_dash_boundary(self):
        self.assertEqual(self.p.resolve("gpt-5.5-2026-04-23"), "gpt-5.5")
        self.assertEqual(self.p.resolve("gpt-5.1-codex-mini"), "gpt-5.1-codex-mini")
        self.assertEqual(self.p.resolve("gpt-5.6-sol"), "gpt-5.6-sol")
        self.assertEqual(self.p.resolve("gpt-5.7"), self.p.default_model)
        self.assertIn("gpt-5.7", self.p.unknown_models)

    def test_unknown_model_uses_default_and_is_recorded(self):
        self.assertEqual(self.p.resolve("mystery-9"), "gpt-5.5")
        self.assertIn("mystery-9", self.p.unknown_models)

    def test_expensive(self):
        self.assertTrue(self.p.is_expensive("gpt-5.5"))
        self.assertTrue(self.p.is_expensive("gpt-5.6-sol"))
        self.assertFalse(self.p.is_expensive("gpt-5.6-terra"))
        self.assertFalse(self.p.is_expensive("gpt-5.3-codex"))

    def test_cost_addition(self):
        s = Cost(1, 2, 3, 4) + Cost(1, 1, 1, 1)
        self.assertEqual((s.input, s.cache_write, s.cache_read, s.output), (2, 3, 4, 5))
        self.assertEqual(s.total, 14)

    def test_gpt61_terra_default_write_is_30m(self):
        c = self.p.cost("gpt-6.1-terra", Usage(input=10000, cache_write=10000))
        self.assertAlmostEqual(c.cache_write, 10000 * 1.25 * 1e-6)
        self.assertAlmostEqual(c.input, 0.0)
        self.assertEqual(self.p.rates("gpt-6.1-terra")["write_ttl"], "30m")

    def test_cache_write_12h_from_usage_split(self):
        u = Usage(input=12000, cached=4000, cache_write=7000, cache_write_12h=4000)
        c = self.p.cost("gpt-6.1-terra", u)
        self.assertAlmostEqual(c.input, 1000 * 1e-6)
        self.assertAlmostEqual(c.cache_write, (3000 * 1.25 + 4000 * 2.5) * 1e-6)
        self.assertAlmostEqual(c.cache_read, 4000 * 0.05 * 1e-6)

    def test_cache_write_12h_clamped_to_cache_write(self):
        c = self.p.cost("gpt-5.5", Usage(input=1000, cache_write=1000, cache_write_12h=5000))
        self.assertAlmostEqual(c.cache_write, 1000 * 2.5 * 5e-6)

    def test_cache_write_ttl_12h_prices_undifferentiated_writes(self):
        p = Pricing({
            "default_model": "gpt-5.5",
            "cache_write_ttl": "12h",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {"gpt-5.5": {"input": 5, "output": 30}},
        })
        c = p.cost("gpt-5.5", Usage(input=1000, cache_write=1000))
        self.assertAlmostEqual(c.cache_write, 1000 * 2.5 * 5e-6)
        # A real TTL split still wins over the undifferentiated default, including a 0.
        split = p.cost("gpt-5.5", Usage(input=1000, cache_write=1000, cache_write_12h=400))
        self.assertAlmostEqual(split.cache_write, (600 * 1.25 + 400 * 2.5) * 5e-6)
        zero_12h = p.cost("gpt-5.5", Usage(input=1000, cache_write=1000, cache_write_12h=0))
        self.assertAlmostEqual(zero_12h.cache_write, 1000 * 1.25 * 5e-6)

    def test_per_model_cache_write_override(self):
        p = Pricing({
            "default_model": "m",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {"m": {"input": 2, "output": 8, "cache_write": 2.5}},
        })
        c = p.cost("m", Usage(input=1000, cache_write=1000))
        self.assertAlmostEqual(c.cache_write, 1000 * 2.5 * 2e-6)

    def test_per_model_cache_write_ttl_overrides_global(self):
        p = Pricing({
            "default_model": "a",
            "cache_write_ttl": "30m",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25, "cache_write_12h": 2.5},
            "models": {
                "a": {"input": 5, "output": 30},
                "b": {"input": 5, "output": 30, "cache_write_ttl": "12h"},
            },
        })
        self.assertAlmostEqual(p.cost("a", Usage(input=100, cache_write=100)).cache_write, 100 * 1.25 * 5e-6)
        self.assertAlmostEqual(p.cost("b", Usage(input=100, cache_write=100)).cache_write, 100 * 2.5 * 5e-6)

    def test_legacy_pricing_json_without_12h_keys(self):
        data = {
            "default_model": "gpt-5.5",
            "multipliers": {"cache_read": 0.1, "cache_write": 1.25},
            "models": {"gpt-5.5": {"input": 5, "output": 30}},
        }
        p = Pricing(data)
        c = p.cost("gpt-5.5", Usage(input=1000, cache_write=1000))
        self.assertAlmostEqual(c.cache_write, 1000 * 1.25 * 5e-6)
        self.assertEqual(p.cache_write_ttl, "30m")
        p12 = Pricing(dict(data, cache_write_ttl="12h"))
        c12 = p12.cost("gpt-5.5", Usage(input=1000, cache_write=1000))
        self.assertAlmostEqual(c12.cache_write, 1000 * 2.5 * 5e-6)
        split = p.cost("gpt-5.5", Usage(input=1000, cache_write=1000, cache_write_12h=400))
        self.assertAlmostEqual(split.cache_write, (600 * 1.25 + 400 * 2.5) * 5e-6)


if __name__ == "__main__":
    unittest.main()
