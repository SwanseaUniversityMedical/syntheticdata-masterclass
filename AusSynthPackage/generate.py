"""
generate(): synthesise relational data from metadata.

Key design contract:
  - Per-column marginal distributions are matched (sampled from the
    published value_counts or bins).
  - Cross-column relationships *within* a table are NOT preserved
    (this is by design — each column is sampled independently).
  - Referential integrity IS preserved: every foreign key value points to
    a valid synthetic parent primary key.
  - Cardinality (mean child rows per parent) is approximately preserved.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(
    metadata: Union[Dict, str],
    n_rows: Optional[Dict[str, int]] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    level: int = 2,
    smoothing: float = 0.6,
    dp_epsilon: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic tables from a metadata document.

    Parameters
    ----------
    metadata : dict or str
        Either the metadata dict returned by ``process()``, or a path to a
        JSON file produced by ``process()``.
    n_rows : dict, optional
        Override the number of rows per top-level (parent) table.
        Example: ``{"patients": 500}``.
        Child tables size themselves from the recorded cardinality so that
        the mean child-per-parent ratio is preserved.
    output_dir : str, optional
        If given, each synthetic table is written as ``<table>.csv`` here.
    seed : int, optional
        Random seed for reproducibility.
    smoothing : float, default 0.0
        Gaussian within-bin smoothing for continuous numeric columns.  The
        added noise has ``std = smoothing × bin_width``.  ``0.0`` reproduces
        the strict uniform-within-bin behaviour (visible "stained-glass"
        grid on 2D plots like LAT/LON).  ``0.1``–``0.3`` softens bin edges
        for visual continuity while keeping marginals roughly intact.
        Pure generator-side; does not affect metadata or SDC.
    dp_epsilon : float, optional
        Differential-privacy-style noise parameter — **only applied when
        ``level=3``**.  Adds Laplace(scale = 1 / dp_epsilon) noise to every
        histogram count used during sampling: CART tree leaf counts, global
        marginal value_counts used for categorical calibration, and global
        bins used for continuous calibration.  Smaller epsilon = more noise
        = stronger privacy, at the cost of utility.

        Typical range: ``0.1`` (heavy noise, strong privacy) to ``10`` (light
        noise, near-original utility).  ``1.0`` is a common starting point.
        ``None`` (default) disables noise injection entirely.

        Note: this is an *approximate* DP mechanism — the total privacy
        budget across all sampling sites is greater than a single
        ``dp_epsilon`` would suggest under strict composition.  Treat it as
        a tunable privacy/utility knob, not a formal ε-DP guarantee.

    Returns
    -------
    dict
        Mapping ``{table_name: synthetic DataFrame}``.
    """
    if isinstance(metadata, str):
        with open(metadata) as f:
            metadata = json.load(f)

    rng = np.random.default_rng(seed)
    tables_meta = metadata["tables"]

    if dp_epsilon is not None and dp_epsilon <= 0:
        raise ValueError(f"dp_epsilon must be positive, got {dp_epsilon}")
    if dp_epsilon is not None and level < 3:
        # Silently ignored at lower levels — warn so users don't expect privacy.
        import warnings as _w
        _w.warn(
            "dp_epsilon is ignored when level < 3 (noise is only applied to "
            "level-3 conditional tree leaves and marginal calibration).",
            stacklevel=2,
        )

    # 1. Topological order: parents before children
    order = _topological_order(tables_meta)

    # 2. Decide row counts up front
    row_counts = _resolve_row_counts(tables_meta, order, n_rows)

    # 3. Generate, parent tables first
    synthetic: Dict[str, pd.DataFrame] = {}
    for tname in order:
        synthetic[tname] = _generate_table(
            tname, tables_meta[tname], row_counts[tname], synthetic, rng, level,
            tables_meta=tables_meta, smoothing=smoothing, dp_epsilon=dp_epsilon,
        )
        synthetic[tname] = _restore_original_dtypes(synthetic[tname], tables_meta[tname])

    # 4. Optionally persist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for tname, df in synthetic.items():
            df.to_csv(os.path.join(output_dir, f"{tname}.csv"), index=False)

    return synthetic


# ---------------------------------------------------------------------------
# Topology / sizing
# ---------------------------------------------------------------------------

def _topological_order(tables_meta: Dict) -> List[str]:
    """Kahn's algorithm over the parent->child DAG."""
    incoming = {t: set(meta.get("parent_tables", [])) for t, meta in tables_meta.items()}
    order: List[str] = []
    remaining = dict(incoming)

    while remaining:
        # Nodes with no remaining parents
        ready = [t for t, parents in remaining.items() if not parents]
        if not ready:
            # Cycle (or self-reference) — fall back to insertion order for the rest
            ready = list(remaining.keys())
        ready.sort()  # determinism
        for t in ready:
            order.append(t)
            remaining.pop(t)
            for other in remaining.values():
                other.discard(t)
    return order


def _resolve_row_counts(
    tables_meta: Dict,
    order: List[str],
    overrides: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """
    Decide n_rows per table.

    For each table:
      - If the user supplied an override, use it.
      - Else if the table has parent(s), size = sum over parents of
          parent_n_rows * (mean child-per-parent recorded on the parent).
      - Else fall back to the original n_rows from metadata.
    """
    overrides = overrides or {}
    counts: Dict[str, int] = {}
    for tname in order:
        if tname in overrides:
            counts[tname] = int(overrides[tname])
            continue

        meta = tables_meta[tname]
        parents = meta.get("parent_tables", [])
        if not parents:
            counts[tname] = int(meta["n_rows"])
            continue

        # Sum expected children across all parents
        total = 0.0
        used_parent_signal = False
        for parent in parents:
            if parent not in counts:
                # Parent hasn't been sized yet (cycle/self-ref); skip
                continue
            parent_card = tables_meta[parent].get("cardinality", {}).get(tname)
            if parent_card is None:
                continue
            total += counts[parent] * parent_card["mean"]
            used_parent_signal = True

        counts[tname] = int(round(total)) if used_parent_signal else int(meta["n_rows"])
    return counts


# ---------------------------------------------------------------------------
# Per-table generation
# ---------------------------------------------------------------------------

def _generate_table(
    name: str,
    meta: Dict,
    n_rows: int,
    already_built: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    level: int = 2,
    tables_meta: Optional[Dict] = None,
    smoothing: float = 0.0,
    dp_epsilon: Optional[float] = None,
) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame(columns=list(meta["columns"].keys()))

    pk = meta.get("primary_key")
    fk_lookup = {fk["column"]: fk for fk in meta.get("foreign_keys", [])}

    # First, assign FK columns (this also determines the actual row count
    # if FKs are present, because we'll grow rows based on parent cardinality)
    fk_columns_data, actual_n_rows = _assign_fks(
        name, meta, n_rows, already_built, rng, tables_meta=tables_meta,
    )

    # Level 3: conditional generation via CART trees
    if level >= 3 and meta.get("conditional_trees") is not None:
        return _generate_table_conditional(
            meta, actual_n_rows, fk_columns_data, already_built, rng,
            smoothing=smoothing, dp_epsilon=dp_epsilon,
        )

    columns: Dict[str, np.ndarray] = {}

    # Handle linked columns with value pairs
    linked = meta.get("linked_columns", [])
    linked_value_pairs = meta.get("linked_value_pairs", [])
    used_linked_cols = set()
    df = None
    if linked and linked_value_pairs:
        columns = {}
        # For each group, sample tuples and assign to columns
        for group, pairs in zip(linked, linked_value_pairs):
            if not group or not pairs:
                continue
            # Build population and weights
            values = [tuple(pair["values"]) for pair in pairs]
            if level == 1:
                weights = None
            else:
                weights = [pair["count"] for pair in pairs]
            if weights is not None:
                probs = np.array(weights) / np.sum(weights)
                sampled_idxs = rng.choice(len(values), size=actual_n_rows, p=probs)
            else:
                sampled_idxs = rng.choice(len(values), size=actual_n_rows)
            sampled = [values[i] for i in sampled_idxs]
            # Assign to columns
            for idx, col in enumerate(group):
                columns[col] = np.array([row[idx] for row in sampled], dtype=object)
                used_linked_cols.add(col)
        # Fill in the rest as before
        for col, cmeta in meta["columns"].items():
            if col in used_linked_cols:
                continue
            if col == pk:
                columns[col] = _generate_primary_key(col, cmeta, actual_n_rows)
            elif col in fk_columns_data:
                columns[col] = fk_columns_data[col]
            else:
                columns[col] = _sample_column(cmeta, actual_n_rows, rng, level, smoothing=smoothing)
        df = pd.DataFrame(columns)
    else:
        # Fallback: original logic
        columns = {}
        for col, cmeta in meta["columns"].items():
            if col == pk:
                columns[col] = _generate_primary_key(col, cmeta, actual_n_rows)
            elif col in fk_columns_data:
                columns[col] = fk_columns_data[col]
            else:
                columns[col] = _sample_column(cmeta, actual_n_rows, rng, level, smoothing=smoothing)
        df = pd.DataFrame(columns)
    # Enforce linked columns: if any value in a group is None, set all to None
    for group in linked:
        if not group:
            continue
        if not all(col in df.columns for col in group):
            continue
        mask = df[group].isnull().any(axis=1)
        df.loc[mask, group] = None
    # Reorder to match metadata column order
    df = df[list(meta["columns"].keys())]
    return df


# ---------------------------------------------------------------------------
# Primary key generation
# ---------------------------------------------------------------------------

def _generate_primary_key(col: str, cmeta: Dict, n_rows: int) -> np.ndarray:
    dtype = cmeta.get("dtype", "integer")
    if dtype == "integer":
        return np.arange(1, n_rows + 1, dtype=np.int64)
    # Strings / UUIDs / other: produce stable prefixed IDs
    width = max(4, len(str(n_rows)))
    return np.array([f"{col}_{i:0{width}d}" for i in range(1, n_rows + 1)])


# ---------------------------------------------------------------------------
# Foreign key assignment
# ---------------------------------------------------------------------------

def _assign_fks(
    name: str,
    meta: Dict,
    desired_n_rows: int,
    already_built: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    tables_meta: Optional[Dict] = None,
):
    """
    Build the FK columns. Returns (fk_data_dict, actual_n_rows).

    When a single FK is present and the parent's cardinality histogram is
    available, we sample a child count per parent from that distribution so
    the children-per-parent SHAPE (skew, zeros, heavy tails) is preserved —
    not just the mean.  Otherwise we fall back to uniform random sampling.
    """
    fks = meta.get("foreign_keys", [])
    if not fks:
        return {}, desired_n_rows

    fk_data: Dict[str, np.ndarray] = {}

    # Single-FK case with recorded cardinality bins → reproduce the shape
    if len(fks) == 1 and tables_meta is not None:
        fk = fks[0]
        parent_table = fk["references_table"]
        if parent_table in already_built:
            parent_pks = already_built[parent_table][fk["references_column"]].to_numpy()
            if len(parent_pks) > 0:
                card_info = (
                    tables_meta.get(parent_table, {})
                    .get("cardinality", {})
                    .get(name, {})
                )
                bins = card_info.get("bins") or []
                if bins:
                    fk_values = _per_parent_fk_assignment(parent_pks, bins, rng)
                    fk_data[fk["column"]] = fk_values
                    return fk_data, len(fk_values)

    # Fallback (multiple FKs, missing parent, or no cardinality bins):
    # uniform random over parent PKs, sized to desired_n_rows.
    for fk in fks:
        parent_table = fk["references_table"]
        parent_pk_col = fk["references_column"]
        if parent_table not in already_built:
            fk_data[fk["column"]] = rng.integers(1, max(2, desired_n_rows + 1), size=desired_n_rows)
            continue

        parent_pks = already_built[parent_table][parent_pk_col].to_numpy()
        if len(parent_pks) == 0:
            fk_data[fk["column"]] = np.array([], dtype=object)
            continue

        fk_data[fk["column"]] = rng.choice(parent_pks, size=desired_n_rows, replace=True)

    return fk_data, desired_n_rows


def _per_parent_fk_assignment(
    parent_pks: np.ndarray,
    bins: List[Dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each parent PK, draw a child-count from the recorded cardinality
    histogram (bin chosen by count weight, then a uniform integer inside that
    bin).  Returns a flattened array of FK values where parent ``i`` appears
    ``count_i`` times, shuffled so row order doesn't reflect parent order.
    """
    weights = np.array([b["count"] for b in bins], dtype=float)
    if weights.sum() <= 0:
        # Empty / degenerate histogram → uniform sampling fallback
        return rng.choice(parent_pks, size=len(parent_pks), replace=True)
    probs = weights / weights.sum()

    n_parents = len(parent_pks)
    chosen_bins = rng.choice(len(bins), size=n_parents, replace=True, p=probs)

    per_parent_counts = np.empty(n_parents, dtype=np.int64)
    for i, b_idx in enumerate(chosen_bins):
        b = bins[b_idx]
        lo, hi = int(round(float(b["min"]))), int(round(float(b["max"])))
        if hi <= lo:
            per_parent_counts[i] = max(0, lo)
        else:
            per_parent_counts[i] = int(rng.integers(lo, hi + 1))

    fk_values = np.repeat(parent_pks, per_parent_counts)
    rng.shuffle(fk_values)
    return fk_values


# ---------------------------------------------------------------------------
# Column sampling (the marginal-distribution heart of the package)
# ---------------------------------------------------------------------------

def _sample_column(
    cmeta: Dict,
    n_rows: int,
    rng: np.random.Generator,
    level: int = 2,
    smoothing: float = 0.0,
    dp_epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Sample n_rows values for a non-key column from its published marginal.

    Honours:
      - completeness (we inject NaNs at the right rate)
      - value_counts (categorical: sample categories proportional to count,
        '__OTHER__' is replaced by a placeholder string)
      - bins (continuous: pick a bin proportional to count, then uniform
        within the bin; with optional Gaussian within-bin smoothing)
      - dp_epsilon (optional Laplace noise added to histogram counts before
        weighting; see generate() docstring)
    """
    completeness = float(cmeta.get("completeness", 1.0))
    dtype = cmeta.get("dtype", "string")
    is_cat = cmeta.get("is_categorical", False)

    if is_cat:
        vc = _apply_dp_to_value_counts(cmeta.get("value_counts", {}), dp_epsilon, rng)
        values = _sample_categorical(vc, n_rows, dtype, rng, level)
    else:
        noisy_bins = _apply_dp_to_bins(cmeta.get("bins", []), dp_epsilon, rng)
        values = _sample_continuous(
            noisy_bins, n_rows, dtype, rng, level, smoothing=smoothing,
        )

    # Inject missing values to match completeness
    if completeness < 1.0 and n_rows > 0:
        n_missing = int(round((1 - completeness) * n_rows))
        if n_missing > 0:
            # Object array can hold both values and None
            out = np.array(values, dtype=object)
            missing_idx = rng.choice(n_rows, size=min(n_missing, n_rows), replace=False)
            out[missing_idx] = None
            return out

    return values


def _sample_categorical(
    value_counts: Dict[str, int],
    n_rows: int,
    dtype: str,
    rng: np.random.Generator,
    level: int = 2,
) -> np.ndarray:
    if not value_counts or n_rows == 0:
        return np.array([None] * n_rows, dtype=object)

    categories = list(value_counts.keys())
    if level == 1 or not value_counts:
        raw = rng.choice(categories, size=n_rows, replace=True)
    else:
        weights = np.array(list(value_counts.values()), dtype=float)
        probs = weights / weights.sum()
        raw = rng.choice(categories, size=n_rows, replace=True, p=probs)

    # Cast back to native type where possible
    if dtype == "integer":
        return np.array([_safe_int(v) for v in raw], dtype=object)
    if dtype == "float":
        return np.array([_safe_float(v) for v in raw], dtype=object)
    if dtype == "boolean":
        return np.array([_safe_bool(v) for v in raw], dtype=object)
    return raw.astype(object)


def _sample_continuous(
    bins: List[Dict],
    n_rows: int,
    dtype: str,
    rng: np.random.Generator,
    level: int = 2,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Sample n_rows values from a continuous column's bin histogram.

    When ``smoothing > 0`` and the chosen bin has positive width, Gaussian
    noise with ``std = smoothing × bin_width`` is added to the uniform draw.
    This softens the bin-edge discontinuities that produce visible "stained-
    glass" rectangles on 2D plots of jointly-binned columns (e.g. LAT/LON
    at level 3).  Point bins (``hi == lo``, used for zero-inflation peel-off)
    are never smoothed — their value is exact.
    """
    if not bins or n_rows == 0:
        return np.array([None] * n_rows, dtype=object)

    # Level 1: sample uniformly between the overall min and max across bins
    # (user-requested behavior) — do not sample per-bin.
    out = np.empty(n_rows, dtype=object)
    is_datetime = (dtype == "datetime")
    if level == 1:
        # Compute global bounds from bins
        if is_datetime:
            lo_vals = [pd.Timestamp(b["min"]).value for b in bins]
            hi_vals = [pd.Timestamp(b["max"]).value for b in bins]
            lo_all = min(lo_vals)
            hi_all = max(hi_vals)
            for i in range(n_rows):
                if hi_all > lo_all:
                    ts = float(rng.integers(lo_all, hi_all + 1))
                else:
                    ts = float(lo_all)
                out[i] = pd.Timestamp(int(ts)).isoformat()
        else:
            lo_vals = [float(b["min"]) for b in bins]
            hi_vals = [float(b["max"]) for b in bins]
            lo_all = min(lo_vals)
            hi_all = max(hi_vals)
            for i in range(n_rows):
                if hi_all > lo_all:
                    v = rng.uniform(lo_all, hi_all)
                else:
                    v = lo_all
                if dtype == "integer":
                    out[i] = int(round(v))
                else:
                    out[i] = float(v)
        return out

    # Level >1: sample by picking bins (weighted by counts if available)
    counts = np.array([b["count"] for b in bins], dtype=float)
    probs = counts / counts.sum() if counts.sum() > 0 else None
    if probs is not None:
        chosen = rng.choice(len(bins), size=n_rows, replace=True, p=probs)
    else:
        chosen = rng.choice(len(bins), size=n_rows, replace=True)

    out = np.empty(n_rows, dtype=object)
    for i, idx in enumerate(chosen):
        b = bins[idx]
        if is_datetime:
            lo = pd.Timestamp(b["min"]).value  # ns since epoch
            hi = pd.Timestamp(b["max"]).value
            if hi > lo:
                ts = float(rng.integers(lo, hi + 1))
                if smoothing > 0:
                    ts += rng.normal(0.0, (hi - lo) * smoothing)
            else:
                ts = float(lo)
            out[i] = pd.Timestamp(int(ts)).isoformat()
        else:
            lo, hi = float(b["min"]), float(b["max"])
            if hi > lo:
                v = rng.uniform(lo, hi)
                if smoothing > 0:
                    v += rng.normal(0.0, (hi - lo) * smoothing)
            else:
                v = lo
            if dtype == "integer":
                out[i] = int(round(v))
            else:
                out[i] = float(v)
    return out


def _restore_original_dtypes(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Cast generated columns back to the original pandas dtypes when possible."""
    restored = df.copy()
    for col, cmeta in meta.get("columns", {}).items():
        if col not in restored.columns:
            continue

        original_dtype = cmeta.get("original_dtype")
        if not original_dtype:
            continue

        series = restored[col]

        if original_dtype.startswith("datetime64"):
            restored[col] = pd.to_datetime(series, errors="coerce")
            continue

        if original_dtype in ("bool", "boolean"):
            restored[col] = series.astype("boolean")
            continue

        if original_dtype.startswith(("int", "uint", "Int", "UInt")):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any():
                restored[col] = numeric.astype(_nullable_integer_dtype(original_dtype))
            else:
                restored[col] = numeric.astype(original_dtype)
            continue

        if original_dtype.startswith("float"):
            restored[col] = pd.to_numeric(series, errors="coerce").astype(original_dtype)
            continue

        try:
            restored[col] = series.astype(original_dtype)
        except Exception:
            restored[col] = series

    return restored


def _nullable_integer_dtype(dtype_name: str) -> str:
    """Map a numpy integer dtype name to its pandas nullable equivalent."""
    if dtype_name.startswith(("Int", "UInt")):
        return dtype_name
    if dtype_name.startswith("uint"):
        return "UInt" + dtype_name[4:]
    if dtype_name.startswith("int"):
        return "Int" + dtype_name[3:]
    return "Int64"


# ---------------------------------------------------------------------------
# Small casting helpers
# ---------------------------------------------------------------------------

def _safe_int(v):
    if v == "__OTHER__":
        return v
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return v


def _safe_float(v):
    if v == "__OTHER__":
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _safe_bool(v):
    if v == "__OTHER__":
        return v
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "t"):
        return True
    if s in ("false", "0", "no", "f"):
        return False
    return v


# ---------------------------------------------------------------------------
# Level-3 helpers: conditional generation via CART trees
# ---------------------------------------------------------------------------

def _correct_column_marginal(
    arr: np.ndarray,
    cmeta: Dict,
    n_rows: int,
    rng: np.random.Generator,
    dp_epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Swap a minimal number of values so arr's categorical marginal matches the
    stored value_counts.  No-op for continuous columns (bin sampling already
    approximates the marginal).

    Two sources of drift are handled:
      1. Values not in the global marginal (e.g. __OTHER__ leaking from a leaf
         that has a locally-suppressed distribution) — all occurrences are swapped.
      2. Global categories that are over/under-represented vs their target count.

    Targets are computed relative to the actual non-null count so that the
    correction does not fight against already-correct missingness injection.

    When ``dp_epsilon`` is set, the global ``value_counts`` are first perturbed
    with Laplace noise so the correction targets respect the same privacy
    budget as the conditional leaves.
    """
    if not cmeta.get("is_categorical", False):
        return arr

    value_counts = cmeta.get("value_counts")
    if not value_counts:
        return arr

    value_counts = _apply_dp_to_value_counts(value_counts, dp_epsilon, rng)

    # Non-null rows only — targets are proportions of non-null values
    n_non_null = int(sum(1 for v in arr if v is not None))
    if n_non_null == 0:
        return arr

    total_weight = float(sum(value_counts.values()))
    target = {
        k: int(round(v / total_weight * n_non_null))
        for k, v in value_counts.items()
    }

    # Build str_value -> list[row_index] for non-null positions
    val_to_idx: Dict[str, List[int]] = {}
    for i, v in enumerate(arr):
        if v is not None:
            val_to_idx.setdefault(str(v), []).append(i)

    swap_pool: List[int] = []
    under_cats: List[str] = []
    under_deficits: List[int] = []

    # Values absent from the global marginal (e.g. __OTHER__ from leaf SDC):
    # every occurrence is excess and should be replaced
    for s, idxs in val_to_idx.items():
        if s not in target:
            swap_pool.extend(idxs)

    # Values in the global marginal: compare actual vs target count
    for cat, tgt in target.items():
        act = len(val_to_idx.get(cat, []))
        if act > tgt:
            excess = act - tgt
            idxs = val_to_idx[cat]
            chosen = rng.choice(idxs, size=min(excess, len(idxs)), replace=False)
            swap_pool.extend(chosen.tolist())
        elif act < tgt:
            under_cats.append(cat)
            under_deficits.append(tgt - act)

    if not swap_pool or not under_cats:
        return arr

    total_deficit = sum(under_deficits)
    n_replace = min(len(swap_pool), total_deficit)
    if n_replace == 0:
        return arr

    probs = np.array(under_deficits, dtype=float)
    probs /= probs.sum()
    new_vals = rng.choice(under_cats, size=n_replace, p=probs)
    swap_idxs = rng.choice(swap_pool, size=n_replace, replace=False)

    arr = arr.copy()
    for idx, val in zip(swap_idxs, new_vals):
        arr[idx] = val
    return arr


def _correct_continuous_marginal(
    arr: np.ndarray,
    cmeta: Dict,
    rng: np.random.Generator,
    dp_epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Reassign continuous values so the column follows the published global
    bins while preserving the row-wise rank structure already induced by the
    conditional sampler.

    This is a monotone, rank-preserving calibration: it tightens the marginal
    shape without reshuffling relationships with other columns.

    When ``dp_epsilon`` is set, the global bin counts are perturbed with
    Laplace noise before being used to draw calibration targets.
    """
    if cmeta.get("is_categorical", False):
        return arr

    bins = cmeta.get("bins")
    if not bins:
        return arr

    dtype = cmeta.get("dtype", "float")
    non_null_idx = [i for i, v in enumerate(arr) if v is not None]
    if len(non_null_idx) < 2:
        return arr

    noisy_bins = _apply_dp_to_bins(bins, dp_epsilon, rng)
    target = _sample_continuous(
        noisy_bins, len(non_null_idx), dtype, rng, level=2, smoothing=0.0
    )
    if len(target) == 0:
        return arr

    def _sort_key(values: np.ndarray) -> np.ndarray:
        if dtype == "datetime":
            ts = pd.to_datetime(pd.Series(values, dtype=object), errors="coerce")
            if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            numeric = ts.astype("datetime64[ns]").astype("int64").astype(float)
            numeric[pd.isna(ts).to_numpy()] = np.inf
            return numeric
        numeric = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(dtype=float)
        numeric[np.isnan(numeric)] = np.inf
        return numeric

    source_vals = np.asarray([arr[i] for i in non_null_idx], dtype=object)
    source_order = np.argsort(_sort_key(source_vals), kind="mergesort")
    target_order = np.argsort(_sort_key(target), kind="mergesort")

    calibrated = arr.copy()
    ordered_source_idx = [non_null_idx[i] for i in source_order]
    ordered_target_vals = [target[i] for i in target_order]
    for idx, val in zip(ordered_source_idx, ordered_target_vals):
        calibrated[idx] = val
    return calibrated


def _apply_dp_to_value_counts(
    value_counts: Optional[Dict],
    epsilon: Optional[float],
    rng: np.random.Generator,
) -> Optional[Dict]:
    """
    Add Laplace(scale = 1 / epsilon) noise to histogram counts.  Negative
    noisy counts are clipped to 0; the dict is returned unchanged if epsilon
    is None/non-positive, the input is empty, or every count would be 0.
    """
    if not epsilon or epsilon <= 0 or not value_counts:
        return value_counts
    scale = 1.0 / float(epsilon)
    noisy: Dict = {}
    for k, v in value_counts.items():
        noisy[k] = max(0.0, float(v) + float(rng.laplace(0.0, scale)))
    if sum(noisy.values()) <= 0:
        return value_counts
    return noisy


def _apply_dp_to_bins(
    bins: Optional[List[Dict]],
    epsilon: Optional[float],
    rng: np.random.Generator,
) -> Optional[List[Dict]]:
    """
    Add Laplace(scale = 1 / epsilon) noise to bin ``count`` fields.  Bin
    boundaries (min/max) are *not* perturbed — only the weights used to
    pick a bin.  Returns the original list if epsilon is None/non-positive,
    the input is empty, or every count would be 0.
    """
    if not epsilon or epsilon <= 0 or not bins:
        return bins
    scale = 1.0 / float(epsilon)
    noisy: List[Dict] = []
    for b in bins:
        nb = dict(b)
        nb["count"] = max(0.0, float(b.get("count", 0)) + float(rng.laplace(0.0, scale)))
        noisy.append(nb)
    if sum(nb.get("count", 0) for nb in noisy) <= 0:
        return bins
    return noisy


def _generate_table_conditional(
    meta: Dict,
    actual_n_rows: int,
    fk_columns_data: Dict[str, np.ndarray],
    already_built: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    smoothing: float = 0.0,
    dp_epsilon: Optional[float] = None,
) -> pd.DataFrame:
    """Generate a table using sequential CART-conditional column sampling."""
    pk = meta.get("primary_key")
    col_order: List[str] = meta.get("column_order", [])
    conditional_trees: Dict = meta.get("conditional_trees", {})

    # Look up parent attribute arrays for each child row via FK
    parent_context = _build_parent_context(meta, fk_columns_data, already_built)

    result: Dict[str, np.ndarray] = {}

    # Structural columns first (PK and FK values are already determined)
    for col, cmeta in meta["columns"].items():
        if col == pk:
            result[col] = _generate_primary_key(col, cmeta, actual_n_rows)
        elif col in fk_columns_data:
            result[col] = fk_columns_data[col]

    # Content columns in entropy-ranked order so each column can condition
    # on all previously generated ones.  Derived secondary linked columns
    # (e.g. DESCRIPTION given CODE) are skipped here and looked up later.
    for i, col in enumerate(col_order):
        if col in result:
            continue
        cmeta = meta["columns"].get(col)
        if cmeta is None or cmeta.get("derived_from"):
            continue

        if col in conditional_trees:
            vals = _sample_column_conditional(
                conditional_trees[col], cmeta,
                result, parent_context, actual_n_rows, rng,
                smoothing=smoothing, dp_epsilon=dp_epsilon,
            )
            # Correct marginal drift: SDC suppression in small tree leaves can
            # cause rare categories to be under-represented vs the global marginal
            if cmeta.get("is_categorical", False):
                result[col] = _correct_column_marginal(
                    vals, cmeta, actual_n_rows, rng, dp_epsilon=dp_epsilon,
                )
            else:
                result[col] = _correct_continuous_marginal(
                    vals, cmeta, rng, dp_epsilon=dp_epsilon,
                )
        else:
            result[col] = _sample_column(
                cmeta, actual_n_rows, rng, level=2, smoothing=smoothing,
                dp_epsilon=dp_epsilon,
            )

    # Derive secondary linked columns (DESCRIPTION ← CODE, REASONDESCRIPTION ← REASONCODE)
    _derive_linked_secondaries(meta, result, actual_n_rows, rng)

    # Safety fallback for any column still missing
    for col, cmeta in meta["columns"].items():
        if col not in result:
            result[col] = _sample_column(
                cmeta, actual_n_rows, rng, level=2, smoothing=smoothing,
                dp_epsilon=dp_epsilon,
            )

    df = pd.DataFrame(result)[list(meta["columns"].keys())]

    # If any column in a linked group is null in a row, null out the others too
    # (matches the level-2 contract for linked columns).
    for group in meta.get("linked_columns", []):
        if not group:
            continue
        if not all(c in df.columns for c in group):
            continue
        mask = df[group].isnull().any(axis=1)
        df.loc[mask, group] = None

    return df


def _derive_linked_secondaries(
    meta: Dict,
    result: Dict[str, np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
) -> None:
    """
    For each linked group, derive the secondary columns (index 1..N) from the
    primary (index 0) using the recorded ``linked_value_pairs`` as a lookup
    table.  This preserves perfect 1:1 mappings (CODE→DESCRIPTION) without
    spending tree depth or column-order slots on the redundant secondary, and
    leaves the primary free to act as a predictor for other columns' trees.
    """
    linked = meta.get("linked_columns", []) or []
    pairs_per_group = meta.get("linked_value_pairs", []) or []

    for group, pairs in zip(linked, pairs_per_group):
        if len(group) < 2 or not pairs:
            continue
        primary = group[0]
        if primary not in result:
            # Primary wasn't generated (e.g. excluded from col_order); skip.
            continue

        primary_vals = result[primary]

        # Build mapping primary_value -> {secondary_value: count} for each secondary
        for sec_idx, sec_col in enumerate(group[1:], start=1):
            mapping: Dict[str, Dict[object, int]] = {}
            for pair in pairs:
                pv = pair["values"][0]
                sv = pair["values"][sec_idx]
                if pv is None or (isinstance(pv, float) and pd.isna(pv)):
                    continue
                key = str(pv)
                mapping.setdefault(key, {})
                mapping[key][sv] = mapping[key].get(sv, 0) + int(pair["count"])

            out = np.empty(n_rows, dtype=object)
            for i, pv in enumerate(primary_vals):
                if pv is None or (isinstance(pv, float) and pd.isna(pv)):
                    out[i] = None
                    continue
                opts = mapping.get(str(pv))
                if not opts:
                    # Primary value not seen in real tuples (typically __OTHER__);
                    # mirror the marker into the secondary so the row stays
                    # internally consistent.
                    out[i] = pv
                    continue
                if len(opts) == 1:
                    out[i] = next(iter(opts))
                else:
                    keys = list(opts.keys())
                    weights = np.array(list(opts.values()), dtype=float)
                    out[i] = keys[int(rng.choice(len(keys), p=weights / weights.sum()))]
            result[sec_col] = out


def _build_parent_context(
    meta: Dict,
    fk_columns_data: Dict[str, np.ndarray],
    already_built: Dict[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """
    For each FK relationship, look up the parent row for every child row and
    return a dict of {parent__<col>: array_of_length_n_rows} arrays.
    """
    context: Dict[str, np.ndarray] = {}
    parent_context_cols: List[str] = meta.get("parent_context_columns", [])
    if not parent_context_cols:
        return context

    for fk in meta.get("foreign_keys", []):
        parent_name = fk["references_table"]
        fk_col = fk["column"]
        ref_col = fk["references_column"]

        if parent_name not in already_built or fk_col not in fk_columns_data:
            continue

        parent_df = already_built[parent_name]
        if ref_col not in parent_df.columns:
            continue

        # Which parent columns are actually needed?
        needed = [
            c[len("parent__"):]
            for c in parent_context_cols
            if c.startswith("parent__") and c[len("parent__"):] in parent_df.columns
        ]
        if not needed:
            continue

        fk_series = pd.Series(fk_columns_data[fk_col])
        lookup = (
            parent_df
            .drop_duplicates(subset=[ref_col])
            .set_index(ref_col)[needed]
        )

        for col in needed:
            prefixed = f"parent__{col}"
            if prefixed in parent_context_cols:
                context[prefixed] = fk_series.map(lookup[col]).to_numpy()

    return context


def _sample_column_conditional(
    tree_info: Dict,
    cmeta: Dict,
    already_generated: Dict[str, np.ndarray],
    parent_context: Dict[str, np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
    smoothing: float = 0.0,
    dp_epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Sample n_rows values by traversing a CART conditional tree.

    Uses vectorised boolean-mask splits instead of per-row traversal:
    each split partitions the current active set of row indices into left
    and right subsets, which are filled recursively.
    """
    tree = tree_info["tree"]
    predictor_dtypes: Dict[str, str] = tree_info.get("predictor_dtypes", {})
    context = {**already_generated, **parent_context}
    out = np.empty(n_rows, dtype=object)

    def to_numeric_for_compare(feat_arr: np.ndarray, feat_name: str) -> np.ndarray:
        """Convert a context array to the same float space the tree was fitted in."""
        if predictor_dtypes.get(feat_name) == "datetime":
            ts = pd.to_datetime(pd.Series(feat_arr, dtype=object), errors="coerce")
            if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            valid = ts.notna()
            arr = np.full(len(feat_arr), np.inf, dtype=float)
            if valid.any():
                arr[valid.values] = (
                    ts[valid].astype("datetime64[ns]").astype("int64").astype(float) / 1e9
                )
            return arr
        return (
            pd.to_numeric(pd.Series(feat_arr, dtype=object), errors="coerce")
            .fillna(np.inf)
            .to_numpy()
        )

    def fill(node: Dict, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n == 0:
            return

        if node["type"] == "leaf":
            if "value_counts" in node:
                vc = _apply_dp_to_value_counts(node["value_counts"], dp_epsilon, rng)
                out[mask] = _sample_categorical(
                    vc, n, cmeta.get("dtype", "string"), rng, level=2
                )
            else:
                bins = _apply_dp_to_bins(node["bins"], dp_epsilon, rng)
                out[mask] = _sample_continuous(
                    bins, n, cmeta.get("dtype", "float"), rng, level=2,
                    smoothing=smoothing,
                )
            return

        feat = node["feature"]
        feat_vals = context.get(feat)

        if feat_vals is None or len(feat_vals) == 0:
            # Conditioning feature not available: use marginal for this partition
            out[mask] = _sample_column(
                cmeta, n, rng, level=2, smoothing=smoothing, dp_epsilon=dp_epsilon,
            )
            return

        feat_arr = np.asarray(feat_vals, dtype=object)

        if node.get("is_categorical", False):
            cats_left = set(str(c) for c in node.get("categories_left", []))
            left_mask = mask & np.array(
                [str(v) in cats_left if v is not None else False for v in feat_arr],
                dtype=bool,
            )
        else:
            threshold = node["threshold"]
            numeric = to_numeric_for_compare(feat_arr, feat)
            left_mask = mask & (numeric <= threshold)

        right_mask = mask & ~left_mask
        fill(node["left"], left_mask)
        fill(node["right"], right_mask)

    fill(tree, np.ones(n_rows, dtype=bool))

    # Inject missingness to match recorded completeness
    completeness = float(cmeta.get("completeness", 1.0))
    if completeness < 1.0 and n_rows > 0:
        n_missing = int(round((1.0 - completeness) * n_rows))
        if n_missing > 0:
            missing_idx = rng.choice(n_rows, size=min(n_missing, n_rows), replace=False)
            out[missing_idx] = None

    return out
