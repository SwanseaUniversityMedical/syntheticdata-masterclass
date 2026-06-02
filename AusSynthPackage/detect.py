"""
Detection heuristics for primary keys, foreign keys, data types,
and categorical-vs-continuous classification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data type detection
# ---------------------------------------------------------------------------

def detect_dtype(series: pd.Series) -> str:
    """
    Return a coarse, JSON-friendly data type label.
    One of: 'integer', 'float', 'boolean', 'datetime', 'string'.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "string"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        # If all floats are actually integers (e.g. 1.0, 2.0), treat as integer
        if np.all(np.equal(np.mod(non_null.values, 1), 0)):
            return "integer"
        return "float"

    # Try to parse object/string columns as datetime
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        sample = non_null.head(50).astype(str)
        if _looks_like_datetime(sample):
            return "datetime"

    return "string"


def _looks_like_datetime(sample: pd.Series) -> bool:
    """Best-effort: try parsing a sample as datetime, succeed if >=80% parse."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
        except Exception:
            return False
    return parsed.notna().mean() >= 0.8


# ---------------------------------------------------------------------------
# Categorical vs continuous
# ---------------------------------------------------------------------------

def is_categorical(
    series: pd.Series,
    dtype: str,
    max_unique: int = 20,
    max_unique_ratio: float = 0.05,
) -> bool:
    """
    Decide whether a column should be treated as categorical.

    Rules:
      - strings and booleans are categorical
      - datetimes are not categorical (handled separately as continuous)
      - numeric columns are categorical if they have few unique values
        relative to row count
    """
    if dtype in ("string", "boolean"):
        return True
    if dtype == "datetime":
        return False

    non_null = series.dropna()
    if non_null.empty:
        return True

    n_unique = non_null.nunique()
    n_total = len(non_null)
    unique_threshold = max(max_unique, int(n_total * max_unique_ratio))
    # Use improved logic: categorical if unique values below threshold
    if n_unique < unique_threshold:
        return True
    return False


# ---------------------------------------------------------------------------
# Primary key detection
# ---------------------------------------------------------------------------

# Common naming patterns that hint at an ID column
_ID_HINT_RE = re.compile(r"(^id$|_id$|^id_|key$|^key_|_key$|uuid|guid)", re.IGNORECASE)


def detect_primary_key(df: pd.DataFrame, table_name: str) -> Optional[str]:
    """
    Pick the most likely primary key column.

    Strategy:
      1. Find all columns that are fully unique and non-null.
      2. Prefer ones whose name suggests an ID, especially
         '<table>_id' or 'id' or '<table>id'.
      3. Otherwise return the first uniquely-valued column, or None.
    """
    candidates: List[str] = []
    for col in df.columns:
        s = df[col]
        if s.isna().any():
            continue
        if s.nunique() == len(s):
            candidates.append(col)

    if not candidates:
        return None

    # Stable singular form: drop a trailing 's' if present
    singular = table_name[:-1] if table_name.endswith("s") else table_name

    # Tier 1: exact match for <table>_id / <singular>_id / id
    tier1 = [
        f"{table_name}_id", f"{singular}_id",
        f"{table_name}id", f"{singular}id",
        "id",
    ]
    for name in tier1:
        for c in candidates:
            if c.lower() == name.lower():
                return c

    # Tier 2: any candidate whose name looks ID-ish
    for c in candidates:
        if _ID_HINT_RE.search(c):
            return c

    # Tier 3: fall back to the first unique column
    return candidates[0]


# ---------------------------------------------------------------------------
# Foreign key detection
# ---------------------------------------------------------------------------

def detect_foreign_keys(
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, Optional[str]],
    overlap_threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    For each table, identify columns that look like foreign keys pointing at
    another table's primary key.

    Returns a dict:
        {table_name: [{"column": ..., "references_table": ..., "references_column": ...}, ...]}

    Heuristics combined (any one is sufficient):
      A. Name match: column name == parent PK name, or contains the singular
         parent name plus 'id'.
      B. Value overlap: >=overlap_threshold of non-null values in the child
         column appear in the parent PK column, AND dtype is compatible.

    The table's own primary key is never reported as a foreign key.
    """
    fks: Dict[str, List[Dict]] = {t: [] for t in tables}

    for child_name, child_df in tables.items():
        child_pk = primary_keys.get(child_name)

        for col in child_df.columns:
            if col == child_pk:
                continue

            child_vals = child_df[col].dropna()
            if child_vals.empty:
                continue

            best_match: Optional[Tuple[str, str, float]] = None  # (parent, parent_pk, score)

            for parent_name, parent_df in tables.items():
                if parent_name == child_name:
                    continue
                parent_pk = primary_keys.get(parent_name)
                if parent_pk is None:
                    continue

                score = _fk_match_score(
                    col, child_vals,
                    parent_name, parent_df[parent_pk],
                    overlap_threshold,
                )
                if score > 0 and (best_match is None or score > best_match[2]):
                    best_match = (parent_name, parent_pk, score)

            if best_match is not None:
                fks[child_name].append({
                    "column": col,
                    "references_table": best_match[0],
                    "references_column": best_match[1],
                })

    return fks


def _fk_match_score(
    child_col: str,
    child_vals: pd.Series,
    parent_table: str,
    parent_pk_vals: pd.Series,
    overlap_threshold: float,
) -> float:
    """
    Combined score for FK candidacy. Returns 0.0 if not a plausible FK.
    Higher = better.

    Philosophy
    ----------
    We are deliberately conservative: false-positive FKs corrupt the
    inferred topology, the cardinality stats, and (downstream) the
    generator. We therefore REQUIRE a name signal — pure value overlap
    is too easy to trigger on incidental integer-range overlap (ages,
    doses, scores...) and is treated only as a tie-breaker.
    """
    # dtype compatibility check (both numeric, or both string-like)
    child_numeric = pd.api.types.is_numeric_dtype(child_vals)
    parent_numeric = pd.api.types.is_numeric_dtype(parent_pk_vals)
    if child_numeric != parent_numeric:
        return 0.0

    # --- Name-based signal (REQUIRED) ---
    singular = parent_table[:-1] if parent_table.endswith("s") else parent_table
    child_lower = child_col.lower()

    exact_name_candidates = {
        f"{parent_table}_id".lower(), f"{singular}_id".lower(),
        f"{parent_table}id".lower(), f"{singular}id".lower(),
    }

    name_score = 0.0
    if child_lower in exact_name_candidates:
        name_score = 1.0
    elif (
        singular.lower() in child_lower
        and ("id" in child_lower or "key" in child_lower)
        and len(singular) >= 3  # avoid spurious 1–2 letter matches
    ):
        name_score = 0.6

    if name_score == 0.0:
        return 0.0  # no name evidence -> not an FK

    # --- Value overlap (used to confirm the name signal) ---
    parent_set = set(parent_pk_vals.unique())
    overlap = float(child_vals.isin(parent_set).mean())

    if overlap < overlap_threshold:
        # The name suggests an FK but the values don't agree.
        # Don't claim it; better to leave it as a plain column.
        return 0.0

    return name_score + overlap
