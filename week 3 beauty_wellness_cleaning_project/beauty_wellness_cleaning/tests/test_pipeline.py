"""Unit + integration tests.  Run with:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bw_pipeline import cleaning as cl                      # noqa: E402
from bw_pipeline import config as C                         # noqa: E402
from bw_pipeline import outliers as ol                      # noqa: E402
from bw_pipeline.data_generator import generate_raw_data    # noqa: E402
from bw_pipeline.imputation import reconcile_by_entity      # noqa: E402
from bw_pipeline.pipeline import Pipeline, build_ml_dataset  # noqa: E402
from bw_pipeline.validation import validate_clean, validate_ml  # noqa: E402


def S(*v):
    return pd.Series(list(v), dtype="string")


class TestParsers(unittest.TestCase):
    def test_price_formats_and_currency(self):
        p, cur = cl.parse_price(S("₹1,299", "Rs. 999", "INR 1,299.00", "499 /-", "$10.00", "€10.00", "abc"))
        self.assertEqual(p.iloc[:4].tolist(), [1299, 999, 1299, 499])
        self.assertAlmostEqual(p.iloc[4], 830.0); self.assertAlmostEqual(p.iloc[5], 900.0)
        self.assertTrue(np.isnan(p.iloc[6])); self.assertEqual(cur.iloc[4], "USD")

    def test_size_units(self):
        v, u = cl.parse_size(S("50 ml", "50ml", "1.7 fl oz", "0.1 L", "100gm", "3.5 oz", "0.1 kg", "junk"))
        self.assertEqual(u.iloc[:4].tolist(), ["ml"] * 4)
        self.assertAlmostEqual(v.iloc[2], 50.3, places=1); self.assertEqual(v.iloc[3], 100)
        self.assertEqual(u.iloc[4], "g"); self.assertAlmostEqual(v.iloc[5], 99.2, places=1); self.assertEqual(v.iloc[6], 100)
        self.assertTrue(np.isnan(v.iloc[7]))

    def test_rating_scales_words_typos(self):
        r, _ = cl.parse_rating(S("4", "4.0", "4 stars", "4/5", "8/10", "four", "55", "0", "6", "-1"))
        self.assertEqual(r.iloc[:7].tolist(), [4, 4, 4, 4, 4, 4, 5])
        self.assertTrue(r.iloc[7:].isna().all())

    def test_mixed_dates(self):
        d = cl.parse_mixed_dates(S("2024-03-05", "05/03/2024", "05-03-2024", "05 Mar 2024", "Mar 05, 2024", "05.03.2024", "garbage"))
        self.assertTrue((d.iloc[:6] == pd.Timestamp("2024-03-05")).all())   # day-first for ambiguous values
        self.assertTrue(pd.isna(d.iloc[6]))

    def test_spf(self):
        s = cl.parse_spf(S("SPF 30", "50+", "0", "SPF 500", "none"))
        self.assertEqual(s.iloc[:3].tolist(), [30, 50, 0]); self.assertTrue(s.iloc[3:].isna().all())


class TestStandardisation(unittest.TestCase):
    def test_missing_tokens(self):
        df, n = cl.normalize_missing_tokens(pd.DataFrame({"a": ["x", "N/A", " ? ", "", "null"]}))
        self.assertEqual(df["a"].isna().sum(), 4); self.assertGreaterEqual(n, 3)

    def test_category_exact_synonym_fuzzy(self):
        out, st = cl.standardize_labels(S("SKINCARE", "skin care", "Skincre", " Hair-Care ", "zzz", None), C.CATEGORIES, C.CATEGORY_SYNONYMS)
        self.assertEqual(out.iloc[:4].tolist(), ["Skincare", "Skincare", "Skincare", "Haircare"])
        self.assertTrue(pd.isna(out.iloc[4]) and pd.isna(out.iloc[5])); self.assertEqual(st["unmatched"], 1)

    def test_city_synonyms(self):
        out, _ = cl.standardize_labels(S("CBE", "kovai", "Bangalore", "Madras"), C.CITIES, C.CITY_SYNONYMS)
        self.assertEqual(out.tolist(), ["Coimbatore", "Coimbatore", "Bengaluru", "Chennai"])

    def test_yes_no(self):
        self.assertEqual(cl.standardize_yes_no(S("Y", "TRUE", "1", "no", "0", "maybe")).tolist(), ["Yes", "Yes", "Yes", "No", "No", pd.NA])


class TestTextAndPII(unittest.TestCase):
    def test_review_text(self):
        self.assertEqual(cl.clean_review_text("<p>LOVE   it!!!!! 😍 https://x.co/a</p>"), "LOVE it!")
        self.assertEqual(cl.clean_review_text("GREAT PRODUCT!!!"), "Great product!")   # all-caps is un-shouted
        self.assertTrue(pd.isna(cl.clean_review_text("   ")))

    def test_ingredients(self):
        self.assertEqual(cl.normalize_ingredients("Aqua; Vitamin C | glycerine, Water (Aqua)"), "water|ascorbic acid|glycerin")

    def test_email_phone_hash(self):
        e, ok = cl.clean_email(S("A@B.com", "john@@gmail", "x"))
        self.assertEqual(e.iloc[0], "a@b.com"); self.assertEqual(ok.tolist(), [True, False, False])
        ph = cl.clean_phone(S("+91 98765-43210", "098765 43210", "12345"))
        self.assertEqual(ph.iloc[0], "9876543210"); self.assertEqual(ph.iloc[1], "9876543210"); self.assertTrue(pd.isna(ph.iloc[2]))
        h = cl.pseudonymize(S("a", "a", "b"))
        self.assertEqual(h.iloc[0], h.iloc[1]); self.assertNotEqual(h.iloc[0], h.iloc[2])


class TestOutliersAndImputation(unittest.TestCase):
    def test_iqr_and_mad_flag_extreme(self):
        s = pd.Series([10, 11, 12, 10, 11, 9, 10, 500.0])
        self.assertTrue(ol.detect_outliers_iqr(s).iloc[-1]); self.assertTrue(ol.detect_outliers_mad(s).iloc[-1])
        self.assertFalse(ol.detect_outliers_iqr(s).iloc[:-1].any())

    def test_reconcile_by_entity(self):
        df = pd.DataFrame({"pid": ["a", "a", "a", "b"], "brand": pd.array(["X", pd.NA, "Y", "Z"], dtype="string")})
        out, rep = reconcile_by_entity(df, "pid", ["brand"])
        self.assertEqual(out["brand"].iloc[1], out["brand"].iloc[0]); self.assertEqual(rep["brand"]["filled"], 1)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate_raw_data(1500, seed=11)
        cls.pipe = Pipeline()
        cls.clean = cls.pipe.run(cls.raw)

    def test_all_validation_checks_pass(self):
        failed = [c for c in validate_clean(self.clean) if not c["passed"]]
        self.assertEqual(failed, [])

    def test_ml_matrix(self):
        ml, cols = build_ml_dataset(self.clean)
        self.assertEqual([c for c in validate_ml(ml, cols) if not c["passed"]], [])
        self.assertAlmostEqual((ml["split"] == "test").mean(), C.TEST_SIZE, delta=0.01)

    def test_rows_accounted_for(self):
        q = sum(len(x) for x in self.pipe.ctx.quarantine)
        dup = sum(v["count"] for k, v in self.pipe.ctx.issues.items() if k.startswith("duplicates:"))
        self.assertEqual(len(self.raw), len(self.clean) + q + dup)

    def test_deterministic(self):
        again = Pipeline().run(generate_raw_data(1500, seed=11))
        pd.testing.assert_frame_equal(self.clean, again)

    def test_pii_removed(self):
        self.assertNotIn("customer_email", self.clean); self.assertNotIn("customer_phone", self.clean)
        self.assertTrue(self.clean["email_hash"].dropna().str.len().eq(16).all())


if __name__ == "__main__":
    unittest.main()
