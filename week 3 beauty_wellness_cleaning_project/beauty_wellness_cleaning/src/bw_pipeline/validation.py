"""Rule-based data validation (a light-weight stand-in for Great Expectations / pandera)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _check(name: str, passed: bool, detail: str = "") -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def validate_clean(df: pd.DataFrame) -> list[dict]:
    lo_r, hi_r = C.RATING_RANGE
    lo_a, hi_a = C.AGE_RANGE
    checks = [
        _check("required columns present", all(c in df for c in C.REQUIRED_COLUMNS)),
        _check("no nulls in required columns", int(df[C.REQUIRED_COLUMNS].isna().sum().sum()) == 0),
        _check("review_id unique", df["review_id"].is_unique, f"{int(df['review_id'].duplicated().sum())} duplicates"),
        _check("no duplicate (customer, product, date)", not df.duplicated(["customer_id", "product_id", "review_date"]).any()),
        _check("rating within 1-5", df["rating"].between(lo_r, hi_r).all()),
        _check("age within 13-90", df["age"].between(lo_a, hi_a).all()),
        _check("price positive and below hard cap", df["price_inr"].between(C.PRICE_MIN_INR, C.PRICE_HARD_MAX_INR).all()),
        _check("price within category ceiling (winsorised)", (df["price_inr_winsor"] <= df["category"].map(C.PRICE_MAX_BY_CATEGORY)).all()),
        _check("size within plausible range", df["size_value"].between(*C.SIZE_RANGE).all()),
        _check("spf within 0-100", df["spf"].between(*C.SPF_RANGE).all()),
        _check("category canonical", df["category"].isin(C.CATEGORIES).all()),
        _check("brand canonical or Unknown", df["brand"].isin(C.BRANDS + ["Unknown"]).all()),
        _check("skin_type canonical or Unknown", df["skin_type"].isin(C.SKIN_TYPES + ["Unknown"]).all()),
        _check("city canonical or Unknown", df["city"].isin(C.CITIES + ["Unknown"]).all()),
        _check("review_date not in the future", (df["review_date"] <= C.REFERENCE_DATE).all()),
        _check("review_date on/after launch_date", ((df["review_date"] >= df["launch_date"]) | df["launch_date"].isna()).all()),
        _check("no raw PII columns", not any(c in df for c in ["customer_email", "customer_phone"])),
        _check("no infinite values", not np.isinf(df.select_dtypes("number")).any().any()),
    ]
    return checks


def validate_ml(ml: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    train = ml[ml["split"] == "train"]
    scaled = [c for c in feature_cols if c.endswith("_scaled")]
    return [
        _check("no NaN in model matrix", int(ml[feature_cols].isna().sum().sum()) == 0),
        _check("no inf in model matrix", not np.isinf(ml[feature_cols]).any().any()),
        _check("train/test split present", set(ml["split"]) == {"train", "test"}),
        _check("standard-scaled train means ~0", all(abs(train[c].mean()) < 0.05 for c in scaled if c.endswith("_std_scaled")),
               "scalers fitted on TRAIN only (no leakage)"),
    ]


def quality_scorecard(before: dict, after: dict) -> pd.DataFrame:
    rows = [{"dimension": k, "before": round(before[k] * 100, 1), "after": round(after[k] * 100, 1)} for k in before]
    rows.append({"dimension": "overall", "before": round(np.mean(list(before.values())) * 100, 1),
                 "after": round(np.mean(list(after.values())) * 100, 1)})
    return pd.DataFrame(rows)
