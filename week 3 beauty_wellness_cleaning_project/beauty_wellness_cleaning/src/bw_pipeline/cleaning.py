"""Reusable cleaning primitives: tokens, parsers, standardisers, text and PII handling.

All functions are vectorised (pandas ``.str`` accessors / ``map``) and free of
side effects on their inputs, which makes them easy to unit-test.
"""
from __future__ import annotations

import difflib
import hashlib
import html
import re
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Column names / missing tokens
# --------------------------------------------------------------------------- #
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """'Customer Age ' -> 'customer_age' (snake_case)."""
    out = df.copy()
    out.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") for c in out.columns]
    return out


def normalize_missing_tokens(df: pd.DataFrame, tokens: Iterable[str] = C.MISSING_TOKENS) -> tuple[pd.DataFrame, int]:
    """Trim whitespace and convert placeholder strings ('N/A', '?', '-', ...) to real missing values."""
    out = df.copy()
    tokens = set(tokens)
    replaced = 0
    for col in out.columns:
        s = out[col].astype("string").str.strip()
        mask = s.str.lower().isin(tokens) | s.isna()
        replaced += int((mask & out[col].notna() & (out[col].astype("string").str.strip() != "")).sum())
        s = s.mask(mask, pd.NA)
        out[col] = s
    return out, replaced


# --------------------------------------------------------------------------- #
# Categorical standardisation (exact -> synonym -> fuzzy)
# --------------------------------------------------------------------------- #
def _key(x: str) -> str:
    x = str(x).lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def standardize_labels(series: pd.Series, canonical: list[str], synonyms: dict[str, str] | None = None,
                       cutoff: float = C.FUZZY_CUTOFF) -> tuple[pd.Series, dict[str, int]]:
    """Map messy labels onto a canonical vocabulary.

    1. normalise (case, punctuation, whitespace)  2. exact / synonym lookup
    3. fuzzy match (difflib ratio >= cutoff) for typos like 'Skincre'.
    Returns the cleaned Series and counters {'exact','synonym','fuzzy','unmatched'}.
    """
    lookup = {_key(c): c for c in canonical}
    syn = {_key(k): v for k, v in (synonyms or {}).items()}
    stats = {"exact": 0, "synonym": 0, "fuzzy": 0, "unmatched": 0}
    cache: dict[str, object] = {}

    def resolve(val):
        if pd.isna(val):
            return pd.NA
        k = _key(val)
        if k in cache:
            return cache[k]
        if k in lookup:
            res, how = lookup[k], "exact"
        elif k in syn:
            res, how = syn[k], "synonym"
        else:
            close = difflib.get_close_matches(k, list(lookup) + list(syn), n=1, cutoff=cutoff)
            if close:
                res = lookup.get(close[0], syn.get(close[0]))
                how = "fuzzy"
            else:
                res, how = pd.NA, "unmatched"
        cache[k] = (res, how)[0]
        cache[k + "__how"] = how
        return res

    out = series.map(resolve)
    for v in series.dropna():
        resolve(v)
        stats[cache[_key(v) + "__how"]] += 1
    # count only values that actually changed (exact matches with identical text are not "fixes")
    changed = int(((out.astype("string") != series.astype("string")) & series.notna()).sum())
    stats["changed"] = changed
    return out.astype("string"), stats


def standardize_yes_no(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[s.isin(C.YES_TOKENS)] = "Yes"
    out[s.isin(C.NO_TOKENS)] = "No"
    return out


# --------------------------------------------------------------------------- #
# Numeric parsers
# --------------------------------------------------------------------------- #
_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_price(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'₹1,299', 'Rs. 999', 'INR 1,299.00', '$12.99', '499 /-' -> float INR (+ detected currency)."""
    s = series.astype("string").str.lower()
    cur = pd.Series("INR", index=s.index, dtype="string")
    cur[s.str.contains(r"\$|usd", na=False)] = "USD"
    cur[s.str.contains(r"€|eur", na=False)] = "EUR"
    num = pd.to_numeric(s.str.replace(",", "", regex=False).str.extract(_NUM)[0], errors="coerce").astype(float)
    rate = cur.map(C.FX_TO_INR).astype(float)
    return (num * rate).round(2), cur.where(series.notna(), pd.NA)


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|ml|ltr|litre|liter|kg|gms|gm|oz|l|g)\b")


def parse_size(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'1.7 fl oz', '50ml', '0.1 L', '100gm' -> (value in ml or g, base unit)."""
    values, units = [], []
    for raw in series.astype("string"):
        if pd.isna(raw):
            values.append(np.nan); units.append(pd.NA); continue
        m = _SIZE_RE.search(str(raw).lower())
        neg = str(raw).strip().startswith("-")
        if not m:
            values.append(np.nan); units.append(pd.NA); continue
        unit_raw = re.sub(r"[\s.]+", " ", m.group(2)).strip()
        base, factor = C.UNIT_TO_BASE[unit_raw]
        v = float(m.group(1)) * factor
        values.append(-v if neg else v); units.append(base)
    return pd.Series(values, index=series.index, dtype=float).round(1), pd.Series(units, index=series.index, dtype="string")


def parse_rating(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Ratings such as '4', '4.0', '4 stars', '4/5', '8/10', 'four', '55' -> float on a 1-5 scale.

    Returns (rating, corrected_flag).  Repeated-digit values ('44', '55') are treated
    as double-key typos and corrected to the single digit; anything else outside 1-5
    becomes missing.
    """
    t = series.astype("string").str.lower().str.strip()
    for word, val in C.WORD_NUMBERS.items():
        t = t.str.replace(rf"\b{word}\b", str(val), regex=True)
    frac = t.str.extract(rf"{_NUM}\s*/\s*{_NUM}")
    num, den = pd.to_numeric(frac[0], errors="coerce").astype(float), pd.to_numeric(frac[1], errors="coerce").astype(float)
    val = pd.to_numeric(t.str.extract(_NUM)[0], errors="coerce").astype(float)
    has_frac = num.notna() & den.notna() & (den > 0)
    val = val.where(~has_frac, num / den * 5)
    typo = val.isin([11, 22, 33, 44, 55])
    val = val.where(~typo, val % 10)
    lo, hi = C.RATING_RANGE
    val = val.where((val >= lo) & (val <= hi))
    val = (val * 2).round() / 2          # snap to nearest 0.5
    return val, typo | has_frac


def parse_spf(series: pd.Series) -> pd.Series:
    """'SPF 30', '50+', 'SPF 50 PA++++' -> int 0-100 (NaN if absent / invalid)."""
    v = pd.to_numeric(series.astype("string").str.extract(r"(\d{1,3})")[0], errors="coerce").astype(float)
    lo, hi = C.SPF_RANGE
    return v.where((v >= lo) & (v <= hi))


def parse_mixed_dates(series: pd.Series, formats: list[str] = C.DATE_FORMATS) -> pd.Series:
    """Parse a column containing several date formats; ambiguous 05/03/2024 is read DAY-first."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce").astype("datetime64[ns]")
        out = out.fillna(parsed)
    return out


# --------------------------------------------------------------------------- #
# Text, ingredients, contacts
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")


def clean_review_text(text):
    """Strip HTML, URLs, emojis; collapse repeated punctuation/whitespace; un-shout ALL CAPS."""
    if pd.isna(text):
        return pd.NA
    t = html.unescape(str(text))
    t = _TAG_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"([!?.])\1{1,}", r"\1", t)
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t.isupper():
        t = t.lower().capitalize()
    return t if t else pd.NA


def normalize_ingredients(text) -> str | float:
    """Split on , ; | newline -> lower-case -> synonym map -> de-duplicate (order kept) -> 'a|b|c'."""
    if pd.isna(text):
        return pd.NA
    parts = [p.strip().strip(".").lower() for p in re.split(r"[,;|\n]+", str(text))]
    out, seen = [], set()
    for p in parts:
        if not p:
            continue
        p = C.INGREDIENT_SYNONYMS.get(p, p)
        if p not in seen:
            seen.add(p); out.append(p)
    return "|".join(out) if out else pd.NA


def ingredient_flags(ingredients: pd.Series) -> pd.DataFrame:
    """Boolean claim/safety flags + ingredient count, derived from the normalised list."""
    s = ingredients.astype("string").fillna("")
    flags = {name: s.apply(lambda x, keys=keys: int(any(k in x.split("|") for k in keys))) for name, keys in C.INGREDIENT_FLAGS.items()}
    out = pd.DataFrame(flags, index=ingredients.index)
    out["n_ingredients"] = s.apply(lambda x: len([i for i in x.split("|") if i]))
    return out


_EMAIL_RE = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")


def clean_email(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype("string").str.strip().str.lower()
    valid = s.str.match(_EMAIL_RE).fillna(False).astype(bool)
    return s.where(valid, pd.NA), valid & series.notna()


def clean_phone(series: pd.Series) -> pd.Series:
    """Keep the last 10 digits of an Indian mobile number; invalid numbers become missing."""
    digits = series.astype("string").str.replace(r"\D", "", regex=True)
    digits = digits.where(digits.str.len() >= 10).str[-10:]
    return digits.where(digits.str.match(r"^[6-9]\d{9}$").fillna(False), pd.NA)


def pseudonymize(series: pd.Series, salt: str = C.PII_SALT) -> pd.Series:
    """Salted SHA-256 (first 16 hex chars): joins/duplicate checks still work, raw PII is gone."""
    return series.map(lambda x: hashlib.sha256((salt + str(x)).encode()).hexdigest()[:16] if pd.notna(x) else pd.NA).astype("string")
