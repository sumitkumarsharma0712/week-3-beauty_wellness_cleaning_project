#!/usr/bin/env python3
"""Run the Beauty & Wellness cleaning pipeline end-to-end.

Usage:
    python main.py                       # generate synthetic raw data (5,000 rows) and clean it
    python main.py --input my_raw.csv    # clean your own raw CSV (same column layout)
    python main.py --rows 20000 --seed 7 --strict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bw_pipeline import config as C                                    # noqa: E402
from bw_pipeline import visualization as viz                           # noqa: E402
from bw_pipeline.data_generator import write_raw_csv                   # noqa: E402
from bw_pipeline.pipeline import (DATA_DICTIONARY, Pipeline, build_ml_dataset, run_manifest)  # noqa: E402
from bw_pipeline.validation import quality_scorecard, validate_clean, validate_ml           # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="raw CSV to clean (default: generate synthetic data)")
    ap.add_argument("--rows", type=int, default=5000, help="rows to generate when no --input is given")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if any validation check fails")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    for d in (C.DATA_RAW, C.DATA_INTERIM, C.DATA_PROCESSED, C.REPORTS_DIR, C.FIGURES_DIR, C.LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(C.LOGS_DIR / "pipeline.log", mode="w", encoding="utf-8")])
    log = logging.getLogger("bw_pipeline")

    # ---------------------------------------------------------------- 1. ingest
    if args.input:
        raw_path = args.input
    else:
        raw_path = C.RAW_FILE
        write_raw_csv(raw_path, n_rows=args.rows, seed=args.seed)
        log.info("Generated synthetic raw data -> %s", raw_path)
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)      # read EVERYTHING as text
    log.info("Loaded raw data: %d rows x %d columns", *raw.shape)

    # ---------------------------------------------------------------- 2. clean
    pipe = Pipeline()
    clean = pipe.run(raw)
    ctx = pipe.ctx
    quarantine = pd.concat(ctx.quarantine, ignore_index=True) if ctx.quarantine else pd.DataFrame()
    ml, feature_cols = build_ml_dataset(clean, seed=args.seed)

    # ---------------------------------------------------------------- 3. validate
    checks = validate_clean(clean) + validate_ml(ml, feature_cols)
    failed = [c for c in checks if not c["passed"]]
    for c in checks:
        log.info("CHECK %-48s %s", c["check"], "PASS" if c["passed"] else f"FAIL {c['detail']}")

    # ---------------------------------------------------------------- 4. scorecard
    key_cols = [c for c in ["age", "gender", "city", "skin_type", "skin_concern", "price_inr", "size_value", "brand", "launch_date",
                            "spf", "ingredients_clean", "rating", "would_recommend", "purchase_channel", "review_date"] if c in clean]
    miss_before = ctx.snapshots["missing_before"]
    raw_names = {"price_inr": "price", "size_value": "size", "ingredients_clean": "ingredients"}
    comp_before = 1 - np.mean([miss_before.get(raw_names.get(c, c), 0) for c in key_cols]) / 100
    n_raw = ctx.snapshots["raw_rows"]
    before = {"completeness": comp_before, "uniqueness": 1 - (raw.shape[0] - len(clean) - len(quarantine)) / n_raw,
              "validity": 1 - ctx.snapshots["rows_with_invalid_values"] / n_raw, "consistency": ctx.snapshots["consistency_before"]}
    # AFTER completeness is strict: 'Unknown' labels and imputed values do NOT count as complete
    strict = pd.DataFrame({c: (clean[c].isna() | (clean[c] == "Unknown")) for c in key_cols if c not in ("launch_date", "age", "size_value")})
    strict["age"] = clean["age_imputed"].astype(bool)
    strict["size_value"] = clean["size_value_imputed"].astype(bool)
    after = {"completeness": 1 - strict.mean().mean(),
             "uniqueness": 1.0 - clean.duplicated(["customer_id", "product_id", "review_date"]).mean(),
             "validity": float(np.mean([c["passed"] for c in checks])), "consistency": 1.0}
    score = quality_scorecard(before, after)

    # ---------------------------------------------------------------- 5. write outputs
    outs = {"clean": C.DATA_PROCESSED / "beauty_wellness_clean.csv", "ml_ready": C.DATA_PROCESSED / "beauty_wellness_ml_ready.csv",
            "quarantine": C.DATA_PROCESSED / "quarantine_rows.csv"}
    export = clean.drop(columns=["recommend_flag"], errors="ignore")
    export.to_csv(outs["clean"], index=False)
    ml.to_csv(outs["ml_ready"], index=False)
    quarantine.to_csv(outs["quarantine"], index=False)
    log_df = pd.DataFrame(ctx.step_log)
    issues = pd.DataFrame(ctx.issues.values()).query("count > 0").reset_index(drop=True)
    log_df.to_csv(C.REPORTS_DIR / "cleaning_log.csv", index=False)
    issues.to_csv(C.REPORTS_DIR / "issue_register.csv", index=False)
    score.to_csv(C.REPORTS_DIR / "quality_scorecard.csv", index=False)
    pd.DataFrame([{"column": c, "dtype": str(export[c].dtype), "description": DATA_DICTIONARY.get(c, "")} for c in export.columns]
                 ).to_csv(C.REPORTS_DIR / "data_dictionary.csv", index=False)
    pd.DataFrame(checks).to_csv(C.REPORTS_DIR / "validation_checks.csv", index=False)
    summary = {"raw_rows": int(n_raw), "clean_rows": int(len(clean)), "quarantined_rows": int(len(quarantine)),
               "exact_duplicates_removed": int(ctx.snapshots["exact_dups"]), "ml_features": len(feature_cols),
               "checks_passed": int(len(checks) - len(failed)), "checks_total": len(checks),
               "scorecard": score.set_index("dimension").to_dict("index"),
               "quarantine_reasons": quarantine["quarantine_reason"].value_counts().to_dict() if len(quarantine) else {}}
    (C.REPORTS_DIR / "quality_report.json").write_text(json.dumps(summary, indent=2))
    (C.REPORTS_DIR / "run_manifest.json").write_text(json.dumps(run_manifest(raw_path, outs, int(n_raw), len(clean), args.seed), indent=2))

    # ---------------------------------------------------------------- 6. figures
    if not args.no_plots:
        viz.plot_flowchart()
        after_missing = (clean[[c for c in clean.columns if c in miss_before.index or c in raw_names.values() or True]].isna().mean() * 100)
        after_missing.index = [{"price_inr": "price", "size_value": "size", "ingredients_clean": "ingredients"}.get(c, c) for c in after_missing.index]
        viz.plot_missing(miss_before.drop(["review_id", "customer_id", "product_id", "product_name"], errors="ignore"),
                         after_missing.drop(["review_id", "customer_id", "product_id", "product_name"], errors="ignore"))
        viz.plot_price(ctx.snapshots["price_before"], clean[["category", "price_inr_winsor"]].rename(columns={"price_inr_winsor": "price_inr"}))
        viz.plot_rating(ctx.snapshots["rating_before"], clean["rating"])
        viz.plot_categories(ctx.snapshots["category_before"], clean["category"].value_counts())
        viz.plot_scorecard(score)
        viz.plot_pipeline_attrition(log_df)

    log.info("DONE: %d raw -> %d clean (+%d quarantined) | checks %d/%d passed | quality %.1f%% -> %.1f%%",
             n_raw, len(clean), len(quarantine), len(checks) - len(failed), len(checks),
             score.iloc[-1]["before"], score.iloc[-1]["after"])
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
