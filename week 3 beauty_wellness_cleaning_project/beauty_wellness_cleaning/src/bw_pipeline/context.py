"""Run context: collects the issue register, step log and 'before' snapshots."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger("bw_pipeline")


@dataclass
class PipelineContext:
    issues: dict[str, dict[str, Any]] = field(default_factory=dict)   # issue register
    step_log: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, Any] = field(default_factory=dict)           # "before" data for charts
    quarantine: list[pd.DataFrame] = field(default_factory=list)      # rejected rows with reasons

    def record(self, step: str, issue: str, count: int, action: str) -> None:
        """Register how many records were affected by `issue` and what was done."""
        count = int(count)
        self.issues[f"{step}:{issue}"] = {"step": step, "issue": issue, "count": count, "action": action}
        if count:
            logger.info("[%s] %s -> %d record(s): %s", step, issue, count, action)

    def reject(self, df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        """Move rows flagged by `mask` into quarantine (kept for audit) and return the rest."""
        mask = mask.fillna(False)
        if mask.any():
            q = df.loc[mask].copy()
            q["quarantine_reason"] = reason
            self.quarantine.append(q)
            self.record("quarantine", reason, int(mask.sum()), "rows removed from analysis set, saved for audit")
        return df.loc[~mask].copy()
