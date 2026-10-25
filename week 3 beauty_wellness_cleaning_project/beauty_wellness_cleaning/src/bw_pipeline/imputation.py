"""Missing-value strategies.

Order of preference (most to least trustworthy):
  1. Entity reconciliation  - the same product/customer appears in many rows; the most
     frequent non-null value is the best estimate for a missing/conflicting one.
  2. Group-wise statistics  - median (numeric) or mode (categorical) of a peer group.
  3. Global constant        - 'Unknown' / 0, always accompanied by an *_imputed flag
     where the information may carry signal (missing-not-at-random).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mode_or_na(x: pd.Series):
    x = x.dropna()
    return x.mode().iloc[0] if len(x) else np.nan


def reconcile_by_entity(df: pd.DataFrame, entity_col: str, cols: list[str], overwrite_conflicts: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Fill gaps (and optionally repair conflicts) using the per-entity most frequent value."""
    out = df.copy()
    report: dict[str, dict[str, int]] = {}
    for col in cols:
        if col not in out:
            continue
        mode = out.groupby(entity_col)[col].agg(_mode_or_na)
        target = out[entity_col].map(mode).astype(out[col].dtype)
        missing = out[col].isna() & target.notna()
        conflict = out[col].notna() & target.notna() & (out[col] != target)
        out[col] = out[col].where(~missing, target)
        if overwrite_conflicts:
            out[col] = out[col].where(~conflict, target)
        report[col] = {"filled": int(missing.sum()), "conflicts_repaired": int(conflict.sum()) if overwrite_conflicts else 0}
    return out, report


def impute_group_median(df: pd.DataFrame, col: str, group_cols: list[str]) -> tuple[pd.Series, int]:
    """Median of the peer group; falls back to the global median for empty groups."""
    s = df[col].astype(float)
    group_median = s.groupby([df[g] for g in group_cols]).transform("median")
    filled = s.fillna(group_median).fillna(s.median())
    return filled, int(s.isna().sum())


def impute_constant(s: pd.Series, value) -> tuple[pd.Series, int]:
    n = int(s.isna().sum())
    return s.fillna(value), n


def missingness_by_group(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
    """% missing of `col` per level of `group_col` - evidence for MAR vs MCAR (shown in the report)."""
    return (df[col].isna().groupby(df[group_col]).mean() * 100).round(1).sort_values(ascending=False)
