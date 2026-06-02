"""
Statistical Disclosure Control (SDC) utilities.

Any bin or category in the published metadata must have a count >=
the configured threshold (default 10). This module implements:

  - categorical suppression (drop categories below threshold,
    optionally aggregate the remainder into '__OTHER__')
  - continuous binning via quantile bins (each bin holds equal count
    by construction, so SDC safety doesn't depend on a separate merge
    pass), with a zero-inflation peel-off for single-point spikes at 0
  - datetime binning (treated as continuous on the timestamp axis)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def categorical_value_counts(
    series: pd.Series,
    threshold: int = 10,
    aggregate_remainder: bool = True,
) -> Dict[str, int]:
    """
    Build SDC-safe value counts for a categorical column.

    Categories with count < threshold are dropped. If aggregate_remainder is
    True and the total of dropped categories is itself >= threshold, they
    are combined into an '__OTHER__' bucket; otherwise they are suppressed
    entirely.

    Returns: dict mapping category string -> count.
    """
    non_null = series.dropna()
    if non_null.empty:
        return {}

    # Counts as plain Python strings/ints so it's JSON-serialisable
    raw = non_null.astype(str).value_counts()

    safe = raw[raw >= threshold]
    dropped = raw[raw < threshold]

    result = {str(k): int(v) for k, v in safe.items()}

    if aggregate_remainder and not dropped.empty:
        other_total = int(dropped.sum())
        if other_total >= threshold:
            result["__OTHER__"] = other_total

    return result


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------

def continuous_bins(
    series: pd.Series,
    threshold: int = 10,
    n_bins: int = 10,
    is_datetime: bool = False,
) -> List[Dict]:
    """
    Build SDC-safe quantile bins for a continuous (or datetime) column.

    Strategy
    --------
    Quantile bins are used as the single strategy.  Each bin holds roughly
    the same COUNT of values rather than the same WIDTH, so bin edges
    automatically follow data density: narrow where data is dense, wide
    where data is sparse.  This handles heavy tails, multi-modality, and
    non-uniform shapes correctly with no separate heuristics, and is
    SDC-safe by construction (each bin's count is ~``n_total / n_bins``).

    A single pre-processing step is retained: when a substantial fraction
    of values are exactly zero (a structural spike that no continuous bin
    can represent cleanly), zeros are peeled off into a dedicated [0, 0]
    bin before quantile binning the non-zero tail.

    Returns a list of dicts in order, lowest bin first:
        {"min": <num/str>, "max": <num/str>, "count": <int>}

    For datetimes, 'min' and 'max' are ISO 8601 strings; otherwise floats.
    Returns [] when the column is too sparse to publish even one bin.
    """
    non_null = series.dropna()
    if len(non_null) < threshold:
        return []  # whole column too sparse to safely publish

    # 1. Convert to numeric (or int64 nanoseconds for datetimes)
    if is_datetime:
        ts = pd.to_datetime(non_null)
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        ts = ts.astype("datetime64[ns]")
        values = ts.astype("int64").to_numpy()
    else:
        values = pd.to_numeric(non_null, errors="coerce").dropna().to_numpy()
        if len(values) < threshold:
            return []

    # 2. Zero-inflation peel-off (numeric only; datetime "zero" = 1970 isn't
    #    a structural spike).  When >=5% of values are exactly zero and the
    #    zero count meets SDC, give them their own [0, 0] bin so the rest
    #    of the bins aren't distorted by the spike.
    zero_bin: Optional[Dict] = None
    if not is_datetime:
        zero_count = int((values == 0).sum())
        if zero_count >= threshold and zero_count / len(values) >= 0.05:
            zero_bin = {"min": 0.0, "max": 0.0, "count": zero_count}
            values = values[values != 0]
            if len(values) < threshold:
                return [zero_bin]

    # 3. Decide the bin count given the SDC constraint.  Each quantile bin
    #    holds ~n_total/n_bins rows; cap n_bins so that ratio stays >=
    #    threshold.
    n_effective = max(1, min(n_bins, len(values) // threshold))

    # 4. Quantile edges, deduplicated.  Ties at percentile boundaries (e.g.
    #    a value spike) collapse to a single edge, which gracefully reduces
    #    the bin count.
    edges = np.unique(
        np.quantile(values.astype(float), np.linspace(0.0, 1.0, n_effective + 1))
    )

    if len(edges) < 2:
        # All remaining values identical (e.g. another spike)
        only = {
            "min": _format_edge(float(edges[0]), is_datetime),
            "max": _format_edge(float(edges[0]), is_datetime),
            "count": int(len(values)),
        }
        return ([zero_bin, only] if zero_bin else [only])

    # 5. Count values in each bin (right-exclusive except for the final bin)
    counts, _ = np.histogram(values, bins=edges)

    out: List[Dict] = []
    for i, c in enumerate(counts):
        if c < threshold:
            # Theoretically rare with quantile bins, but possible when many
            # ties shift mass into a neighbouring bin.  Skip to keep the
            # published metadata SDC-safe.
            continue
        out.append({
            "min": _format_edge(float(edges[i]), is_datetime),
            "max": _format_edge(float(edges[i + 1]), is_datetime),
            "count": int(c),
        })

    if not out:
        return [zero_bin] if zero_bin else []
    if zero_bin is not None:
        out.insert(0, zero_bin)
    return out


def _format_edge(value: float, is_datetime: bool):
    if is_datetime:
        # Don't go via float rounding — that destroys nanosecond precision.
        return pd.Timestamp(int(round(value))).isoformat()
    # Round to a sensible precision to keep metadata tidy
    return float(np.round(value, 6))
