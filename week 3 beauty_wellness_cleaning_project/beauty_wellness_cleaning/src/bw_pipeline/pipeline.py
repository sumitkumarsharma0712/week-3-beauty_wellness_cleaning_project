"""Orchestration of the cleaning pipeline.

Each step is a small function ``step(df, ctx) -> df``.  The :class:`Pipeline`
runs them in order and logs rows in/out, missing cells and run time per step
(``cleaning_log.csv``) so every transformation is auditable and repeatable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import cleaning as cl
from . import config as C
from . import outliers as ol
from .context import PipelineContext
from .features import add_features
from .imputation import impute_constant, impute_group_median, reconcile_by_entity
from .validation import quality_scorecard, validate_clean, validate_ml

logger = logging.getLogger("bw_pipeline")
Step = Callable[[pd.DataFrame, PipelineContext], pd.DataFrame]


# =========================================================================== #
# STEPS
# =========================================================================== #
def s01_standardize_columns(df, ctx):
    df = cl.standardize_column_names(df)
    return df.rename(columns={"customer_age": "age"})


def s02_normalize_missing(df, ctx):
    df, replaced = cl.normalize_missing_tokens(df)
    ctx.record("missing", "placeholder tokens ('N/A', '?', '-', ...)", replaced, "converted to true missing values")
    ctx.snapshots["missing_before"] = (df.isna().mean() * 100).round(2)
    ctx.snapshots["raw_rows"] = len(df)
    ctx.snapshots["category_before"] = df["category"].dropna().value_counts()
    return df


def s03_exact_duplicates(df, ctx):
    n = len(df)
    df = df.drop_duplicates().copy()
    ctx.record("duplicates", "exact duplicate rows", n - len(df), "dropped (kept first)")
    ctx.snapshots["exact_dups"] = n - len(df)
    return df


def s04_standardize_categoricals(df, ctx):
    spec = {"brand": (C.BRANDS, C.BRAND_SYNONYMS), "category": (C.CATEGORIES, C.CATEGORY_SYNONYMS),
            "skin_type": (C.SKIN_TYPES, C.SKIN_TYPE_SYNONYMS), "gender": (C.GENDERS, C.GENDER_SYNONYMS),
            "city": (C.CITIES, C.CITY_SYNONYMS), "purchase_channel": (C.CHANNELS, C.CHANNEL_SYNONYMS),
            "skin_concern": (C.SKIN_CONCERNS, {})}
    total_vals = canonical_vals = 0
    for col, (canon, syn) in spec.items():
        raw = df[col]
        total_vals += int(raw.notna().sum())
        canonical_vals += int(raw.isin(canon).sum())
        df[col], st = cl.standardize_labels(raw, canon, syn)
        ctx.record("categoricals", f"inconsistent labels in '{col}'", st["changed"], "mapped to canonical label (exact/synonym/fuzzy)")
        ctx.record("categoricals", f"unmatched labels in '{col}'", st["unmatched"], "set to missing, recovered later from entity master")
    ctx.snapshots["consistency_before"] = canonical_vals / max(total_vals, 1)
    raw_rec = df["would_recommend"]
    df["would_recommend"] = cl.standardize_yes_no(raw_rec)
    ctx.record("categoricals", "would_recommend written as Y/1/TRUE/...", int((raw_rec.notna() & (raw_rec != df["would_recommend"])).sum()), "mapped to Yes/No")
    return df


def s05_parse_numeric(df, ctx):
    # price ---------------------------------------------------------------
    df["price_inr"], currency = cl.parse_price(df["price"])
    ctx.record("numeric", "prices in USD/EUR", int(currency.isin(["USD", "EUR"]).sum()), f"converted to INR at fixed rates {C.FX_TO_INR}")
    ctx.record("numeric", "unparseable price strings", int((df["price"].notna() & df["price_inr"].isna()).sum()), "set to missing")
    ctx.snapshots["price_before"] = pd.DataFrame({"category": df["category"], "price_inr": df["price_inr"]})
    # size ----------------------------------------------------------------
    df["size_value"], df["size_unit"] = cl.parse_size(df["size"])
    raw_size = df["size"].astype("string")
    ctx.record("numeric", "sizes in mixed units (fl oz, oz, L, kg)", int(raw_size.str.contains(r"fl|oz|\bl\b|\d\s*l\b|kg", case=False, na=False).sum()), "converted to ml / g")
    ctx.record("numeric", "unparseable size strings", int((df["size"].notna() & df["size_value"].isna()).sum()), "set to missing")
    # rating --------------------------------------------------------------
    raw_num = pd.to_numeric(df["rating"].astype("string").str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").astype(float)
    ctx.snapshots["rating_before"] = raw_num
    df["rating"], rating_fixed = cl.parse_rating(df["rating"])
    ctx.record("numeric", "ratings on other scales / words / typos", int(rating_fixed.sum()), "rescaled to 1-5 or typo corrected")
    ctx.record("numeric", "ratings outside 1-5", int((raw_num.notna() & df["rating"].isna()).sum()), "set to missing (row quarantined later)")
    # spf / age ------------------------------------------------------------
    df["spf"] = cl.parse_spf(df["spf"])
    df["age"] = pd.to_numeric(df["age"].astype("string").str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").astype(float)
    return df.drop(columns=["price", "size"])


def s06_parse_dates(df, ctx):
    for col in ["review_date", "launch_date"]:
        raw = df[col]
        df[col] = cl.parse_mixed_dates(raw)
        ctx.record("dates", f"date strings in {len(C.DATE_FORMATS)} possible formats in '{col}'", int(raw.notna().sum()), "parsed to datetime64 (day-first when ambiguous)")
        ctx.record("dates", f"unparseable dates in '{col}'", int((raw.notna() & df[col].isna()).sum()), "set to NaT")
    return df


def s07_clean_text(df, ctx):
    raw = df["review_text"]
    df["review_text_clean"] = raw.map(cl.clean_review_text).astype("string")
    ctx.record("text", "review texts with HTML/URLs/emoji/shouting/whitespace", int((raw.notna() & (raw != df["review_text_clean"])).sum()), "normalised")
    ing_raw = df["ingredients"]
    df["ingredients_clean"] = ing_raw.map(cl.normalize_ingredients).astype("string")
    ctx.record("text", "ingredient lists with mixed separators/synonyms/duplicates", int((ing_raw.notna() & (ing_raw != df["ingredients_clean"])).sum()), "split, lower-cased, INCI-synonym mapped, de-duplicated")
    return df.drop(columns=["review_text", "ingredients"])


def s08_protect_pii(df, ctx):
    email, valid = cl.clean_email(df["customer_email"])
    ctx.record("pii", "invalid e-mail addresses", int((df["customer_email"].notna() & ~valid).sum()), "set to missing")
    phone = cl.clean_phone(df["customer_phone"])
    ctx.record("pii", "invalid phone numbers", int((df["customer_phone"].notna() & phone.isna()).sum()), "set to missing")
    df["email_domain"] = email.str.split("@").str[1]
    df["email_hash"] = cl.pseudonymize(email)
    df["phone_hash"] = cl.pseudonymize(phone)
    ctx.record("pii", "raw e-mail / phone columns", len(df), "replaced by salted SHA-256 pseudonyms (privacy by design)")
    return df.drop(columns=["customer_email", "customer_phone"])


def s09_near_duplicates(df, ctx):
    key = ["customer_id", "product_id", "review_date"]
    has_key = df[key].notna().all(axis=1)
    work = df.assign(_completeness=df.notna().sum(axis=1))
    work = pd.concat([work[has_key].sort_values("_completeness", ascending=False, kind="stable"), work[~has_key]])
    before = len(work)
    work = pd.concat([work[work[key].notna().all(axis=1)].drop_duplicates(key, keep="first"), work[work[key].isna().any(axis=1)]])
    ctx.record("duplicates", "near-duplicates (same customer+product+date, different review_id)", before - len(work), "kept the most complete record")
    return work.drop(columns="_completeness").sort_index()


def s10_domain_rules(df, ctx):
    lo, hi = C.AGE_RANGE
    bad_age = df["age"].notna() & ~df["age"].between(lo, hi)
    bad_size = df["size_value"].notna() & ~df["size_value"].between(*C.SIZE_RANGE)
    bad_price = df["price_inr"].notna() & ~df["price_inr"].between(C.PRICE_MIN_INR, C.PRICE_HARD_MAX_INR)
    future = df["review_date"].notna() & (df["review_date"] > C.REFERENCE_DATE)
    invalid_any = bad_age | bad_size | bad_price | future
    ctx.snapshots["rows_with_invalid_values"] = int(invalid_any.sum())
    ctx.record("validity", f"age outside {lo}-{hi}", bad_age.sum(), "set to missing")
    ctx.record("validity", "size <=0 or implausibly large", bad_size.sum(), "set to missing")
    ctx.record("validity", f"price < Rs.{C.PRICE_MIN_INR:.0f} or > Rs.{C.PRICE_HARD_MAX_INR:,.0f}", bad_price.sum(), "set to missing")
    ctx.record("validity", "review_date in the future", future.sum(), "set to NaT")
    df.loc[bad_age, "age"] = np.nan
    df.loc[bad_size, "size_value"] = np.nan
    df.loc[bad_price, "price_inr"] = np.nan
    df.loc[future, "review_date"] = pd.NaT
    return df


def s11_reconcile_entities(df, ctx):
    # rows without identifiers cannot be linked -> quarantine
    df = ctx.reject(df, df[["review_id", "customer_id", "product_id"]].isna().any(axis=1), "missing identifier")
    prod_cols = ["product_name", "brand", "category", "size_value", "size_unit", "launch_date", "spf", "ingredients_clean"]
    df, rep = reconcile_by_entity(df, "product_id", prod_cols, overwrite_conflicts=True)
    for col, r in rep.items():
        ctx.record("entity-master", f"product attribute '{col}' missing", r["filled"], "filled from other reviews of the same product")
        ctx.record("entity-master", f"product attribute '{col}' conflicting", r["conflicts_repaired"], "replaced by most frequent value")
    cust_cols = ["age", "gender", "city", "skin_type"]
    df, rep = reconcile_by_entity(df, "customer_id", cust_cols, overwrite_conflicts=True)
    for col, r in rep.items():
        ctx.record("entity-master", f"customer attribute '{col}' missing", r["filled"], "filled from the same customer's other reviews")
    # price: compare each row against the product's median price
    med = df.groupby("product_id")["price_inr"].transform("median")
    ratio = df["price_inr"] / med
    bad = ratio.notna() & ((ratio < C.PRICE_PRODUCT_RATIO[0]) | (ratio > C.PRICE_PRODUCT_RATIO[1]))
    ctx.record("entity-master", "price inconsistent with product's median price (x10 / x0.1 errors)", bad.sum(), "set to missing then re-filled")
    df.loc[bad, "price_inr"] = np.nan
    med = df.groupby("product_id")["price_inr"].transform("median")
    ctx.record("entity-master", "price missing", df["price_inr"].isna().sum(), "filled with product median price")
    df["price_inr"] = df["price_inr"].fillna(med)
    # SPF: product name often carries the claim ("... SPF 30")
    from_name = pd.to_numeric(df["product_name"].str.extract(r"spf\s*(\d{1,3})", flags=2)[0], errors="coerce")
    ctx.record("entity-master", "SPF missing", df["spf"].isna().sum(), "taken from product name, else 0 (no SPF claim)")
    df["spf"] = df["spf"].fillna(from_name).fillna(0)
    return df


def s12_quarantine(df, ctx):
    df = ctx.reject(df, df["rating"].isna(), "missing or invalid rating (target variable)")
    df = ctx.reject(df, df["review_date"].isna(), "missing or invalid review_date")
    conflict = df["launch_date"].notna() & (df["review_date"] < df["launch_date"])
    df = ctx.reject(df, conflict, "review_date before product launch_date")
    df = ctx.reject(df, df["category"].isna(), "category unrecoverable")
    return df


def s13_impute_residual(df, ctx):
    df["age_imputed"] = df["age"].isna().astype(int)
    df["age"], n = impute_group_median(df.assign(_g=df["gender"].fillna("Unknown")), "age", ["_g"])
    df["age"] = df["age"].round().astype(int)
    ctx.record("imputation", "age still missing", n, "median age of gender group (flag age_imputed=1)")
    df["size_value_imputed"] = df["size_value"].isna().astype(int)
    df["size_unit"] = df["size_unit"].fillna(df.groupby("category")["size_unit"].transform(lambda s: s.mode().iloc[0] if s.notna().any() else pd.NA)).fillna("ml")
    sz, n = impute_group_median(df, "size_value", ["category", "size_unit"])
    df["size_value"] = sz
    ctx.record("imputation", "size still missing", n, "median of category x unit")
    pr, n = impute_group_median(df, "price_inr", ["category"])
    df["price_inr"] = pr
    ctx.record("imputation", "price still missing", n, "median price of category")
    for col in ["brand", "gender", "city", "skin_type", "skin_concern", "purchase_channel", "would_recommend"]:
        df[col], n = impute_constant(df[col], "Unknown")
        ctx.record("imputation", f"'{col}' still missing", n, "filled with 'Unknown' category (keeps MNAR signal)")
    df["ingredients_clean"], n = impute_constant(df["ingredients_clean"], "unknown")
    ctx.record("imputation", "ingredients still missing", n, "filled with 'unknown'")
    n = int(df["review_text_clean"].isna().sum())
    df["review_text_clean"] = df["review_text_clean"].fillna("")
    ctx.record("imputation", "review text missing", n, "empty string + has_review_text=0 flag (not imputed)")
    return df


def s14_features(df, ctx):
    return add_features(df)


def s15_outliers(df, ctx):
    df["price_outlier_iqr"] = ol.grouped_iqr_flags(df, "price_inr", "category").astype(int)
    df["price_outlier_mad"] = ol.grouped_mad_flags(df, "price_per_100", "category").astype(int)
    df["price_inr_winsor"] = ol.winsorize_iqr_grouped(df, "price_inr", "category")
    ctx.record("outliers", "statistical price outliers (IQR within category)", df["price_outlier_iqr"].sum(), "flagged + winsorised copy kept (original preserved)")
    ctx.record("outliers", "price-per-100 outliers (modified z-score within category)", df["price_outlier_mad"].sum(), "flagged for review")
    df["price_per_100"] = (df["price_inr_winsor"] / df["size_value"] * 100).round(2)
    feats = df.assign(log_price=np.log1p(df["price_inr"]), log_pp100=np.log1p(df["price_per_100"]))
    flag, score = ol.isolation_forest_flags(feats, ["log_price", "log_pp100", "age", "rating", "review_length", "spf"])
    df["multivariate_outlier"] = flag.astype(int)
    df["anomaly_score"] = score.round(4)
    ctx.record("outliers", "multivariate anomalies (Isolation Forest, 1%)", flag.sum(), "flagged for review, not removed")
    ctx.record("outliers", "rating-text sentiment mismatches", df["rating_sentiment_mismatch"].sum(), "flagged (possible fake / mis-keyed ratings)")
    return df


STEPS: list[tuple[str, Step]] = [
    ("01 standardise column names", s01_standardize_columns), ("02 normalise missing tokens", s02_normalize_missing),
    ("03 remove exact duplicates", s03_exact_duplicates), ("04 standardise categoricals", s04_standardize_categoricals),
    ("05 parse numeric fields", s05_parse_numeric), ("06 parse dates", s06_parse_dates),
    ("07 clean text & ingredients", s07_clean_text), ("08 protect PII", s08_protect_pii),
    ("09 remove near-duplicates", s09_near_duplicates), ("10 apply domain rules", s10_domain_rules),
    ("11 reconcile product/customer master", s11_reconcile_entities), ("12 quarantine unusable rows", s12_quarantine),
    ("13 impute residual gaps", s13_impute_residual), ("14 engineer features", s14_features),
    ("15 detect & treat outliers", s15_outliers),
]


class Pipeline:
    def __init__(self, steps: list[tuple[str, Step]] = STEPS):
        self.steps = steps
        self.ctx = PipelineContext()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for name, fn in self.steps:
            rows_in, na_in, t0 = len(df), int(df.isna().sum().sum()), time.perf_counter()
            df = fn(df, self.ctx)
            self.ctx.step_log.append({"step": name, "rows_in": rows_in, "rows_out": len(df), "rows_removed": rows_in - len(df),
                                      "missing_cells_in": na_in, "missing_cells_out": int(df.isna().sum().sum()),
                                      "columns_out": df.shape[1], "seconds": round(time.perf_counter() - t0, 3)})
            logger.info("%-42s rows %5d -> %5d", name, rows_in, len(df))
        return df.reset_index(drop=True)


# =========================================================================== #
# ML-ready export (scaling + encoding)
# =========================================================================== #
STD_COLS = ["age", "price_log", "size_value", "product_age_days", "n_ingredients", "review_word_count"]
ROBUST_COLS = ["price_per_100_log", "review_length"]
ONEHOT_COLS = ["category", "skin_type", "gender", "purchase_channel", "season", "age_group", "price_tier", "skin_concern", "city"]
BINARY_COLS = ["has_review_text", "is_sunscreen", "age_imputed"] + list(C.INGREDIENT_FLAGS)


def build_ml_dataset(clean: pd.DataFrame, seed: int = C.RANDOM_SEED) -> tuple[pd.DataFrame, list[str]]:
    d = clean.copy()
    d["price_log"] = np.log1p(d["price_inr_winsor"])
    d["price_per_100_log"] = np.log1p(d["price_per_100"])
    train_idx, test_idx = train_test_split(d.index, test_size=C.TEST_SIZE, random_state=seed, stratify=d["high_rating"])
    d["split"] = "train"
    d.loc[test_idx, "split"] = "test"
    train = d.loc[train_idx]
    ml = d[["review_id", "split", "rating", "high_rating"]].copy()
    std, rob = StandardScaler().fit(train[STD_COLS]), RobustScaler().fit(train[ROBUST_COLS])   # fit on TRAIN only
    ml[[f"{c}_std_scaled" for c in STD_COLS]] = std.transform(d[STD_COLS])
    ml[[f"{c}_robust_scaled" for c in ROBUST_COLS]] = rob.transform(d[ROBUST_COLS])
    spf_max = max(train["spf"].max(), 1)
    ml["spf_minmax_scaled"] = (d["spf"] / spf_max).clip(0, 1)
    ml["sentiment_score"] = d["sentiment_score"]
    for c in BINARY_COLS:
        ml[c] = d[c].astype(int)
    ml = pd.concat([ml, pd.get_dummies(d[ONEHOT_COLS], dtype=int)], axis=1)
    feature_cols = [c for c in ml.columns if c not in ("review_id", "split", "rating", "high_rating")]
    return ml, feature_cols


# =========================================================================== #
# Reporting helpers
# =========================================================================== #
DATA_DICTIONARY = {
    "review_id": "Unique review identifier", "customer_id": "Pseudonymous customer key", "product_id": "Product key",
    "age": "Customer age in years (13-90), imputed where unavailable (see age_imputed)", "gender": "Female / Male / Other / Unknown",
    "city": "Standardised Indian city", "skin_type": "Oily / Dry / Combination / Normal / Sensitive / Unknown",
    "skin_concern": "Primary concern the product addresses", "review_date": "Review date (datetime)", "launch_date": "Product launch date",
    "product_name": "Marketing name", "brand": "Canonical brand", "category": "Canonical product category",
    "price_inr": "Price in INR (currency-converted, validated)", "price_inr_winsor": "Price capped at within-category IQR fences",
    "size_value": "Pack size in ml or g", "size_unit": "ml (volume) or g (mass)", "price_per_100": "INR per 100 ml/g (winsorised price)",
    "spf": "Sun protection factor claim (0 = none)", "ingredients_clean": "Pipe-separated normalised ingredient list",
    "rating": "Customer rating 1-5", "high_rating": "1 if rating >= 4 (classification target)", "would_recommend": "Yes / No / Unknown",
    "purchase_channel": "Online / Store / App / Unknown", "review_text_clean": "Cleaned review text", "email_hash": "Salted SHA-256 pseudonym of e-mail",
    "phone_hash": "Salted SHA-256 pseudonym of phone", "email_domain": "E-mail provider domain", "sentiment_score": "Lexicon sentiment -1..1",
    "sentiment_label": "negative / neutral / positive", "rating_sentiment_mismatch": "1 if rating and text sentiment disagree",
    "season": "Indian season of the review month", "product_age_days": "Days between launch and review", "age_group": "Age band",
    "price_tier": "Budget / Mid / Premium within category", "price_outlier_iqr": "1 = IQR outlier within category",
    "price_outlier_mad": "1 = modified z-score outlier on price per 100", "multivariate_outlier": "1 = Isolation Forest anomaly",
    "anomaly_score": "Isolation Forest decision score (lower = more anomalous)", "age_imputed": "1 if age was estimated",
    "size_value_imputed": "1 if size was estimated", "n_ingredients": "Count of distinct ingredients",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_manifest(raw_path, out_files: dict[str, Path], n_raw: int, n_clean: int, seed: int) -> dict:
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "random_seed": seed, "reference_date": str(C.REFERENCE_DATE.date()), "config_sha256": sha256_file(Path(C.__file__)),
            "raw_file_sha256": sha256_file(raw_path), "raw_rows": n_raw, "clean_rows": n_clean,
            "outputs": {k: {"path": str(v.relative_to(C.PROJECT_ROOT)), "sha256": sha256_file(v)} for k, v in out_files.items()}}
