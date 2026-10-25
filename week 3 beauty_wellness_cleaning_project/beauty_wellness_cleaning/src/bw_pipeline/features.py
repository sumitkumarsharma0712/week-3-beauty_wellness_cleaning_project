"""Feature engineering that turns clean records into analysis-ready attributes."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .cleaning import ingredient_flags


def sentiment_score(text: str) -> float:
    """Transparent lexicon score in [-1, 1]: (pos - neg) / (pos + neg), 0 when no cue words."""
    if not text:
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in C.POSITIVE_WORDS for w in words)
    neg = sum(w in C.NEGATIVE_WORDS for w in words)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_per_100"] = (out["price_inr"] / out["size_value"] * 100).round(2)
    text = out["review_text_clean"].fillna("")
    out["has_review_text"] = (text.str.len() > 0).astype(int)
    out["review_length"] = text.str.len()
    out["review_word_count"] = text.str.split().str.len().fillna(0).astype(int)
    out["sentiment_score"] = text.map(sentiment_score).round(3)
    out["sentiment_label"] = pd.cut(out["sentiment_score"], [-1.01, -0.2, 0.2, 1.01], labels=["negative", "neutral", "positive"]).astype("string")
    out["rating_sentiment_mismatch"] = (((out["rating"] >= 4) & (out["sentiment_label"] == "negative")) |
                                        ((out["rating"] <= 2) & (out["sentiment_label"] == "positive"))).astype(int)
    out["review_year"] = out["review_date"].dt.year
    out["review_month"] = out["review_date"].dt.month
    out["review_quarter"] = out["review_date"].dt.quarter
    out["season"] = out["review_month"].map(C.SEASON_BY_MONTH)
    out["product_age_days"] = (out["review_date"] - out["launch_date"]).dt.days
    out["product_age_days"] = out["product_age_days"].fillna(out["product_age_days"].median())
    out["age_group"] = pd.cut(out["age"], C.AGE_BINS, labels=C.AGE_LABELS).astype("string")
    out["is_sunscreen"] = (out["spf"] > 0).astype(int)
    out["price_tier"] = (out.groupby("category")["price_inr"]
                         .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Budget", "Mid", "Premium"]))
                         .astype("string"))
    out["high_rating"] = (out["rating"] >= C.HIGH_RATING_THRESHOLD).astype(int)
    out["recommend_flag"] = out["would_recommend"].map({"Yes": 1, "No": 0})
    flags = ingredient_flags(out["ingredients_clean"].replace("unknown", pd.NA))
    for c in flags.columns:
        out[c] = flags[c]
    return out
