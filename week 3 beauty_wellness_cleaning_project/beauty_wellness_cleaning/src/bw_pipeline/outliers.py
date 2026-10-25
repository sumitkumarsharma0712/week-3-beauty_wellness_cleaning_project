"""Outlier detection: IQR, modified z-score (MAD) and multivariate Isolation Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C


def iqr_bounds(s: pd.Series, k: float = C.IQR_K) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def detect_outliers_iqr(s: pd.Series, k: float = C.IQR_K) -> pd.Series:
    lo, hi = iqr_bounds(s.dropna(), k)
    return ((s < lo) | (s > hi)).fillna(False)


def detect_outliers_mad(s: pd.Series, threshold: float = C.MAD_Z_THRESHOLD) -> pd.Series:
    """Modified z-score 0.6745*(x-median)/MAD; robust because median/MAD ignore the outliers themselves."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)
    z = 0.6745 * (s - med) / mad
    return (z.abs() > threshold).fillna(False)


def grouped_iqr_flags(df: pd.DataFrame, value_col: str, group_col: str, log: bool = True, k: float = C.IQR_K) -> pd.Series:
    """Flag outliers *within each group* (a Rs.6000 perfume is normal, a Rs.6000 soap is not)."""
    vals = np.log1p(df[value_col]) if log else df[value_col]
    flags = pd.Series(False, index=df.index)
    for _, idx in df.groupby(group_col).groups.items():
        flags.loc[idx] = detect_outliers_iqr(vals.loc[idx], k)
    return flags


def winsorize_iqr_grouped(df: pd.DataFrame, value_col: str, group_col: str, k: float = C.IQR_K) -> pd.Series:
    """Cap values at the group's IQR fences (computed on log scale, returned on the original scale)."""
    out = df[value_col].astype(float).copy()
    logv = np.log1p(out)
    for _, idx in df.groupby(group_col).groups.items():
        lo, hi = iqr_bounds(logv.loc[idx].dropna(), k)
        out.loc[idx] = np.expm1(logv.loc[idx].clip(lo, hi))
    return out.round(2)


def isolation_forest_flags(df: pd.DataFrame, cols: list[str], contamination: float = C.ISOLATION_FOREST_CONTAMINATION,
                           seed: int = C.RANDOM_SEED) -> tuple[pd.Series, pd.Series]:
    """Multivariate anomaly flag + score (lower score = more anomalous). Used for review, not deletion."""
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    pred = model.fit_predict(X)
    score = pd.Series(model.decision_function(X), index=df.index)
    return pd.Series(pred == -1, index=df.index), score


def grouped_mad_flags(df: pd.DataFrame, value_col: str, group_col: str, threshold: float = C.MAD_Z_THRESHOLD) -> pd.Series:
    """Modified z-score computed within each group (log scale)."""
    vals = np.log1p(df[value_col])
    flags = pd.Series(False, index=df.index)
    for _, idx in df.groupby(group_col).groups.items():
        flags.loc[idx] = detect_outliers_mad(vals.loc[idx], threshold)
    return flags
