"""Automatic quality control of monitor time series."""
from __future__ import annotations

import pandas as pd


def stuck_values(s: pd.Series, min_run: int) -> pd.Series:
    """True for values repeated identically on ``min_run`` or more consecutive steps."""
    block = (s != s.shift()).cumsum()
    run = s.groupby(block).transform("size")
    return (run >= min_run) & s.notna()


def qc_matrix(M: pd.DataFrame, floor: float | None = None, min_run: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a date x station matrix.

    Flags values below ``floor`` (e.g. an 8-hour ozone of 0 ppb, physically
    impossible in the daytime maximum) and runs of identical values (a stuck
    instrument or a repeated fill value). Returns the cleaned matrix and a
    table of flag counts per station.
    """
    low = M.lt(floor) if floor is not None else pd.DataFrame(False, index=M.index, columns=M.columns)
    stuck = M.apply(lambda s: stuck_values(s, min_run))
    flags = (low | stuck) & M.notna()
    report = pd.DataFrame({"below floor": low.sum(), "stuck runs": stuck.sum(), "flagged": flags.sum(),
                           "valid": M.notna().sum()})
    return M.mask(flags), report[report.flagged > 0].sort_values("flagged", ascending=False)
