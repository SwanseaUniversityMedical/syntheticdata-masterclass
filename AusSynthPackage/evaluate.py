"""
evaluate(): SDMetrics-based evaluation of synthetic relational data.

Four sections of metrics are computed:

  - quality     : statistical fidelity via the SDMetrics multi-table
                  QualityReport (column shapes + column-pair trends +
                  cardinality of FK relationships).
  - diagnostic  : structural validity via the SDMetrics multi-table
                  DiagnosticReport (PK uniqueness, FK validity, value ranges,
                  category coverage).
  - utility     : machine-learning detectability per table — a logistic
                  classifier trained to tell real from synthetic; the
                  reported score is `1 - ROC AUC`-style, so HIGHER is BETTER
                  (1.0 = indistinguishable).
  - privacy     : per-table NewRowSynthesis (fraction of synthetic rows that
                  are not exact duplicates of any real row — HIGHER is BETTER)
                  plus a nearest-neighbour distance summary in normalised
                  numeric space (HIGHER = synthetic sits further from real).

Usage:
    from AusSynthPackage import process, generate, evaluate

    meta = process(real_paths, level=3, ...)
    synth = generate(meta, level=3, seed=0)

    results = evaluate(
        real_tables=real_paths,
        synthetic_tables=synth,
        metadata=meta,                  # AusSynth metadata supplies PK/FK
        output_path="evaluation.json",
    )
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .detect import detect_dtype, is_categorical


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def find_suppressed_columns(
    metadata: Union[Dict, str],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """
    Return ``{table: [columns]}`` where SDC has rolled at least ``threshold``
    of the column's value mass into ``__OTHER__``.  Pass the result straight
    to ``evaluate(..., exclude_columns=...)`` to keep quality/utility scores
    from being dragged down by columns that are intentionally redacted.
    """
    if isinstance(metadata, str):
        with open(metadata) as f:
            metadata = json.load(f)
    out: Dict[str, List[str]] = {}
    for tname, tmeta in metadata.get("tables", {}).items():
        cols = []
        for col, cmeta in tmeta.get("columns", {}).items():
            vc = cmeta.get("value_counts") or {}
            total = sum(vc.values())
            if total > 0 and vc.get("__OTHER__", 0) / total >= threshold:
                cols.append(col)
        if cols:
            out[tname] = cols
    return out




def evaluate(
    real_tables: Dict[str, Union[str, pd.DataFrame]],
    synthetic_tables: Dict[str, Union[str, pd.DataFrame]],
    primary_keys: Optional[Dict[str, Optional[str]]] = None,
    foreign_keys: Optional[Dict[str, List[Dict]]] = None,
    metadata: Optional[Union[Dict, str]] = None,
    exclude_columns: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate synthetic relational tables against the real ones.

    Parameters
    ----------
    real_tables, synthetic_tables : dict
        Mapping of ``table_name -> CSV path or pandas DataFrame``.
    primary_keys : dict, optional
        ``table_name -> pk_column``.  Inferred from ``metadata`` if omitted.
    foreign_keys : dict, optional
        ``table_name -> [{"column", "references_table", "references_column"}, ...]``.
        Inferred from ``metadata`` if omitted.
    metadata : dict or str, optional
        An AusSynth metadata document (the dict from ``process()``) or path to
        its JSON file.  Only consulted to populate ``primary_keys`` /
        ``foreign_keys`` if those weren't given.
    exclude_columns : dict, optional
        ``table_name -> [column, ...]``.  Columns to drop from all four
        evaluation sections.  Useful for SDC-suppressed PII (high-cardinality
        free-text columns like FIRST/LAST/ADDRESS) that are intentionally
        replaced with placeholders in the synthetic data — including them
        would unfairly drag quality and utility scores towards 0.
    output_path : str, optional
        If provided, the full results dict is written here as JSON.
    verbose : bool, default True
        Print progress and a one-line summary per section.

    Returns
    -------
    dict
        ``{"quality": ..., "diagnostic": ..., "utility": ..., "privacy": ...}``.
        See the module docstring for what each section contains.
    """
    real = _load_tables(real_tables)
    synth = _load_tables(synthetic_tables)
    _align_columns(real, synth)

    # Inherit PK/FK from AusSynth metadata if not provided directly
    if metadata is not None:
        if isinstance(metadata, str):
            with open(metadata) as f:
                metadata = json.load(f)
        if primary_keys is None:
            primary_keys = {
                t: m.get("primary_key") for t, m in metadata["tables"].items()
            }
        if foreign_keys is None:
            foreign_keys = {
                t: m.get("foreign_keys", []) for t, m in metadata["tables"].items()
            }

    auto_exclude = _auto_suppressed_string_columns(real, primary_keys, foreign_keys)
    if auto_exclude:
        exclude_columns = _merge_exclude_columns(exclude_columns, auto_exclude)
        if verbose:
            for tname, cols in auto_exclude.items():
                print(f"[evaluate] Auto-suppressing free-text columns in {tname}: {', '.join(cols)}")

    if exclude_columns:
        for tname, cols in exclude_columns.items():
            if tname in real:
                drop = [c for c in cols if c in real[tname].columns]
                real[tname] = real[tname].drop(columns=drop)
                synth[tname] = synth[tname].drop(columns=drop)

    primary_keys = {t: primary_keys.get(t) if primary_keys else None for t in real}
    foreign_keys = {t: (foreign_keys or {}).get(t, []) for t in real}

    # Parse datetime-looking string columns into actual datetime dtype so
    # sdmetrics treats them correctly without needing per-column format strings
    _parse_datetimes_in_place(real, synth)

    # Match synth dtypes to real dtypes (synth comes out as object to allow
    # None injection — sdmetrics treats int 123 and float 123.0 as distinct
    # categories, which crushes TVComplement for whole-number-valued columns).
    _harmonize_dtypes_in_place(real, synth)

    sdm_metadata = _build_sdmetrics_metadata(real, primary_keys, foreign_keys)

    results: Dict[str, Any] = {}

    if verbose:
        print("[evaluate] Generating quality report...")
    results["quality"] = _quality(real, synth, sdm_metadata)

    if verbose:
        print("[evaluate] Generating diagnostic report...")
    results["diagnostic"] = _diagnostic(real, synth, sdm_metadata)

    if verbose:
        print("[evaluate] Computing utility (detection) metrics...")
    results["utility"] = _utility(real, synth, sdm_metadata)

    if verbose:
        print("[evaluate] Computing privacy metrics...")
    results["privacy"] = _privacy(real, synth, sdm_metadata, primary_keys, foreign_keys)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)

    if verbose:
        _print_summary(results)
    # Remove the ReferentialIntegrity metric from the returned results so callers
    # don't see that metric (it can cause downstream code to expect DataFrame
    # shapes). This strips any record or detail row where a string field
    # contains 'Referential' (covers variations like 'ReferentialIntegrity').
    def _strip_referential(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = _strip_referential(v)
            return out
        if isinstance(obj, list):
            new_list = []
            for item in obj:
                # If item is a record (dict), check if any string value
                # references the referential metric and skip it.
                if isinstance(item, dict):
                    skip = False
                    for val in item.values():
                        if isinstance(val, str) and "referential" in val.lower():
                            skip = True
                            break
                    if skip:
                        continue
                new_list.append(_strip_referential(item))
            return new_list
        return obj

    results = _strip_referential(results)

    return results


# ---------------------------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------------------------

def _load_tables(
    tables: Dict[str, Union[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, src in tables.items():
        if isinstance(src, pd.DataFrame):
            out[name] = src.copy()
        elif isinstance(src, str):
            out[name] = pd.read_csv(src)
        else:
            raise TypeError(
                f"Table '{name}' must be a CSV path or DataFrame, got {type(src)}"
            )
    return out


def _merge_exclude_columns(
    base: Optional[Dict[str, List[str]]],
    extra: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {k: list(v) for k, v in (base or {}).items()}
    for tname, cols in extra.items():
        merged.setdefault(tname, [])
        for col in cols:
            if col not in merged[tname]:
                merged[tname].append(col)
    return merged


def _auto_suppressed_string_columns(
    tables: Dict[str, pd.DataFrame],
    primary_keys: Optional[Dict[str, Optional[str]]],
    foreign_keys: Optional[Dict[str, List[Dict]]],
    max_unique: int = 20,
) -> Dict[str, List[str]]:
    """
    Detect high-cardinality string columns that behave like free-text / names
    and exclude them from evaluation automatically.

    The heuristic is intentionally conservative: only columns that are typed as
    strings, are not PK/FK columns, and have many distinct values relative to
    their table size are suppressed.
    """
    auto: Dict[str, List[str]] = {}
    primary_keys = primary_keys or {}
    foreign_keys = foreign_keys or {}

    for tname, df in tables.items():
        pk = primary_keys.get(tname)
        fk_cols = {fk["column"] for fk in foreign_keys.get(tname, [])}
        for col in df.columns:
            if col == pk or col in fk_cols:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            dtype = detect_dtype(series)
            if dtype != "string":
                continue

            n_unique = int(series.astype(str).nunique())
            n_total = int(len(series))
            unique_ratio = n_unique / max(1, n_total)

            # Free-text / name-like fields usually have a very high number of
            # distinct values, often close to one per row. Suppress them so the
            # user doesn't need to manually exclude obvious PII columns like
            # FIRST/LAST/ADDRESS.
            if n_unique > max(max_unique * 3, 50) and unique_ratio > 0.05:
                auto.setdefault(tname, []).append(col)

    return auto


def _align_columns(real: Dict[str, pd.DataFrame], synth: Dict[str, pd.DataFrame]) -> None:
    """Ensure both sides have the same set of columns per table (intersection)."""
    common_tables = set(real) & set(synth)
    extra = (set(real) | set(synth)) - common_tables
    if extra:
        raise ValueError(
            f"Real and synthetic table names must match. Missing on one side: {sorted(extra)}"
        )
    for t in common_tables:
        common_cols = [c for c in real[t].columns if c in synth[t].columns]
        real[t] = real[t][common_cols]
        synth[t] = synth[t][common_cols]


def _parse_datetimes_in_place(
    real: Dict[str, pd.DataFrame],
    synth: Dict[str, pd.DataFrame],
) -> None:
    """Coerce string columns that look like datetimes to actual datetime dtype."""
    for tname, rdf in real.items():
        sdf = synth[tname]
        for col in rdf.columns:
            if pd.api.types.is_datetime64_any_dtype(rdf[col]):
                continue
            if detect_dtype(rdf[col].dropna()) == "datetime":
                rdf[col] = pd.to_datetime(rdf[col], errors="coerce")
                sdf[col] = pd.to_datetime(sdf[col], errors="coerce")


def _harmonize_dtypes_in_place(
    real: Dict[str, pd.DataFrame],
    synth: Dict[str, pd.DataFrame],
) -> None:
    """
    Cast synthetic columns to the same dtype as real where it matters for
    metric comparison.  Handles four cases:

    1. object → numeric  (generator keeps all columns as object to allow None)
    2. object → datetime (same reason)
    3. pandas nullable Int64 → numpy int64 / float64
       The generator uses pandas nullable integer types (Int64) to allow NA.
       sdmetrics' TableStructure sees int64 ≠ Int64 as a dtype mismatch, so
       we cast synth down to the numpy type, using float64 if NAs are present
       (pandas cannot represent NaN in a plain int64 array).
    4. timezone-aware → timezone-naive datetime
       Real CSV dates parsed with UTC timezone; the generator emits tz-naive
       strings.  Strip the timezone from real so both sides are naive.
    """
    for tname, rdf in real.items():
        sdf = synth[tname]
        for col in rdf.columns:
            if col not in sdf.columns:
                continue
            real_dtype = rdf[col].dtype
            synth_dtype = sdf[col].dtype

            # 1. object → numeric
            if pd.api.types.is_numeric_dtype(real_dtype) and not pd.api.types.is_numeric_dtype(synth_dtype):
                sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
                synth_dtype = sdf[col].dtype

            # 2. object → datetime
            elif pd.api.types.is_datetime64_any_dtype(real_dtype) and not pd.api.types.is_datetime64_any_dtype(synth_dtype):
                sdf[col] = pd.to_datetime(sdf[col], errors="coerce")
                synth_dtype = sdf[col].dtype

            # 3. pandas nullable Int64 → numpy int64
            # The generator uses Int64 to hold None; a few NAs may appear even
            # for columns that have no nulls in real (generation artefact).  We
            # fill those NAs from the real column's mode so we can always cast
            # to the numpy dtype, avoiding a float64 fallback that would leave
            # real int64 ≠ synth float64 (a TableStructure mismatch).
            if hasattr(synth_dtype, "numpy_dtype") and not hasattr(real_dtype, "numpy_dtype"):
                target = synth_dtype.numpy_dtype
                if sdf[col].isna().any():
                    mode_vals = rdf[col].mode()
                    fill = mode_vals.iloc[0] if not mode_vals.empty else 0
                    sdf[col] = sdf[col].fillna(fill).astype(target)
                else:
                    sdf[col] = sdf[col].astype(target)
                synth_dtype = sdf[col].dtype

            # 4. Strip timezone from real when synth is tz-naive
            if (
                pd.api.types.is_datetime64_any_dtype(real_dtype)
                and pd.api.types.is_datetime64_any_dtype(synth_dtype)
                and getattr(real_dtype, "tz", None) is not None
                and getattr(synth_dtype, "tz", None) is None
            ):
                rdf[col] = rdf[col].dt.tz_localize(None)


# ---------------------------------------------------------------------------
# SDMetrics metadata construction
# ---------------------------------------------------------------------------

def _sdtype_for(series: pd.Series) -> str:
    """Map a column to an sdmetrics sdtype: id / categorical / numerical / datetime / boolean."""
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    dtype = detect_dtype(series)
    if dtype == "datetime":
        return "datetime"
    if is_categorical(series, dtype, max_unique=20):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    return "categorical"


def _build_sdmetrics_metadata(
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, Optional[str]],
    foreign_keys: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Build an sdmetrics-format multi-table metadata dict."""
    sdm: Dict[str, Any] = {"tables": {}, "relationships": []}

    for tname, df in tables.items():
        pk = primary_keys.get(tname)
        fk_cols = {fk["column"] for fk in foreign_keys.get(tname, [])}
        cols: Dict[str, Dict[str, str]] = {}
        for col in df.columns:
            if col == pk or col in fk_cols:
                cols[col] = {"sdtype": "id"}
            else:
                cols[col] = {"sdtype": _sdtype_for(df[col])}
        entry: Dict[str, Any] = {"columns": cols}
        if pk:
            entry["primary_key"] = pk
        sdm["tables"][tname] = entry

        for fk in foreign_keys.get(tname, []):
            sdm["relationships"].append({
                "parent_table_name": fk["references_table"],
                "parent_primary_key": fk["references_column"],
                "child_table_name": tname,
                "child_foreign_key": fk["column"],
            })

    return sdm


def _single_table_metadata(sdm: Dict[str, Any], tname: str) -> Dict[str, Any]:
    """Extract a single-table metadata dict from the multi-table sdm metadata."""
    entry = sdm["tables"][tname]
    out: Dict[str, Any] = {"columns": dict(entry["columns"])}
    if "primary_key" in entry:
        out["primary_key"] = entry["primary_key"]
    return out


# ---------------------------------------------------------------------------
# Quality + Diagnostic (SDMetrics reports)
# ---------------------------------------------------------------------------

def _quality(real, synth, sdm_metadata) -> Dict[str, Any]:
    try:
        from sdmetrics.reports.multi_table import QualityReport
    except ImportError as e:
        raise ImportError("sdmetrics is required. Install with: pip install sdmetrics") from e

    report = QualityReport()
    report.generate(real, synth, sdm_metadata, verbose=False)

    properties = _df_to_records(report.get_properties())
    details: Dict[str, Any] = {}
    for prop in ("Column Shapes", "Column Pair Trends", "Cardinality", "Intertable Trends"):
        try:
            details[prop] = _df_to_records(report.get_details(property_name=prop))
        except Exception:
            pass

    return {
        "overall_score": float(report.get_score()),
        "properties": properties,
        "details": details,
    }


def _diagnostic(real, synth, sdm_metadata) -> Dict[str, Any]:
    try:
        from sdmetrics.reports.multi_table import DiagnosticReport
    except ImportError as e:
        raise ImportError("sdmetrics is required. Install with: pip install sdmetrics") from e

    report = DiagnosticReport()
    report.generate(real, synth, sdm_metadata, verbose=False)

    properties = _df_to_records(report.get_properties())
    details: Dict[str, Any] = {}
    for prop in ("Data Validity", "Data Structure", "Relationship Validity"):
        try:
            details[prop] = _df_to_records(report.get_details(property_name=prop))
        except Exception:
            pass

    return {
        "overall_score": float(report.get_score()) if hasattr(report, "get_score") else None,
        "properties": properties,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Utility (ML detection — can a classifier tell real from synth?)
# ---------------------------------------------------------------------------

def _utility(real, synth, sdm_metadata) -> Dict[str, Any]:
    """
    For each table, train a logistic regression classifier to distinguish real
    from synthetic and report a detection-based utility score:

        score = 1 - 2 * |ROC_AUC - 0.5|

    so 1.0 means the classifier can't tell real from synthetic (the data is
    highly usable as a substitute) and 0.0 means perfect separation
    (downstream models would learn obviously different things from synth).

    ``overall_score`` is the mean across tables.

    We do not use sdmetrics' LogisticDetection because its OneHotEncoder is
    fitted on real only and fails on categories that exist in synth but not
    real (e.g. ``__OTHER__`` introduced by SDC suppression).
    """
    out: Dict[str, Any] = {}
    table_scores: List[float] = []
    for tname, rdf in real.items():
        sdf = synth[tname]
        single_meta = _single_table_metadata(sdm_metadata, tname)
        try:
            score = _detection_score(rdf, sdf, single_meta)
            out[tname] = {"logistic_detection": score}
            if score is not None:
                table_scores.append(score)
        except Exception as e:
            out[tname] = {"logistic_detection": None, "error": str(e)}
    out["overall_score"] = float(np.mean(table_scores)) if table_scores else None
    return out


def _detection_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    single_meta: Dict[str, Any],
) -> Optional[float]:
    """
    Cross-validated logistic-regression detection score, robust to
    synth-only categories (uses ``handle_unknown='ignore'`` on the encoder
    and fits the encoder on the union of real and synth).
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    cols = single_meta.get("columns", {})
    cat_cols = [
        c for c, s in cols.items()
        if s.get("sdtype") in ("categorical", "boolean")
        and c in real_df.columns and c in synth_df.columns
    ]
    num_cols = [
        c for c, s in cols.items()
        if s.get("sdtype") == "numerical"
        and c in real_df.columns and c in synth_df.columns
    ]
    if not cat_cols and not num_cols:
        return None
    n_real, n_synth = len(real_df), len(synth_df)
    if n_real < 20 or n_synth < 20:
        return None

    rng = np.random.default_rng(0)
    n = min(n_real, n_synth, 2000)
    real_x = real_df.iloc[rng.choice(n_real, size=n, replace=False)].reset_index(drop=True)
    synth_x = synth_df.iloc[rng.choice(n_synth, size=n, replace=False)].reset_index(drop=True)

    parts_real: List[np.ndarray] = []
    parts_synth: List[np.ndarray] = []

    if cat_cols:
        rc = pd.DataFrame({c: _to_python_str_column(real_x[c], "__NULL__") for c in cat_cols})
        sc = pd.DataFrame({c: _to_python_str_column(synth_x[c], "__NULL__") for c in cat_cols})
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # older sklearn
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc.fit(pd.concat([rc, sc], axis=0))
        parts_real.append(enc.transform(rc))
        parts_synth.append(enc.transform(sc))

    if num_cols:
        rn = real_x[num_cols].apply(pd.to_numeric, errors="coerce")
        sn = synth_x[num_cols].apply(pd.to_numeric, errors="coerce")
        meds = rn.median()
        rn = rn.fillna(meds)
        sn = sn.fillna(meds)
        scaler = StandardScaler()
        scaler.fit(pd.concat([rn, sn], axis=0).values)
        parts_real.append(scaler.transform(rn.values))
        parts_synth.append(scaler.transform(sn.values))

    X = np.vstack([np.hstack(parts_real), np.hstack(parts_synth)])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        auc_scores = cross_val_score(
            LogisticRegression(max_iter=500, solver="liblinear"),
            X, y, cv=3, scoring="roc_auc",
        )
    auc = float(np.mean(auc_scores))
    return float(max(0.0, 1.0 - 2.0 * abs(auc - 0.5)))


def _to_python_str_column(series: pd.Series, null_placeholder: str) -> pd.Series:
    """
    Convert a column to genuine Python ``str`` values (not ``numpy.str_``).

    numpy 2.x changed ``repr(np.str_('x'))`` to ``"np.str_('x')"``.  sdmetrics
    embeds those reprs in a ``pd.DataFrame.query()`` expression where the
    name ``np`` is not in scope, so we must coerce to ordinary Python strings
    before handing data off to sdmetrics' single-table metrics.
    """
    values = [null_placeholder if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
              for v in series.tolist()]
    return pd.Series(values, index=series.index, dtype=object)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def _privacy(
    real,
    synth,
    sdm_metadata,
    primary_keys,
    foreign_keys,
) -> Dict[str, Any]:
    try:
        from sdmetrics.single_table import NewRowSynthesis
    except ImportError as e:
        raise ImportError("sdmetrics is required. Install with: pip install sdmetrics") from e

    out: Dict[str, Any] = {}
    for tname, rdf in real.items():
        sdf = synth[tname]
        pk = primary_keys.get(tname)
        fk_cols = [fk["column"] for fk in foreign_keys.get(tname, [])]
        drop = [c for c in [pk] + fk_cols if c]
        rdf_p = rdf.drop(columns=drop, errors="ignore")
        sdf_p = sdf.drop(columns=drop, errors="ignore")

        single_meta = _single_table_metadata(sdm_metadata, tname)
        # Trim metadata to only the columns we kept
        single_meta["columns"] = {
            c: spec for c, spec in single_meta["columns"].items() if c in rdf_p.columns
        }
        single_meta.pop("primary_key", None)

        table_result: Dict[str, Any] = {}

        # NewRowSynthesis uses pd.DataFrame.query() which can't reference np
        # for datetime literals; stringify datetime columns first and mark them
        # categorical for the metric so the query goes via string equality.
        rdf_q, sdf_q, meta_q = _stringify_datetimes(rdf_p, sdf_p, single_meta)

        # Fraction of synthetic rows that don't exactly match any real row.
        # Subsample if synth is large to keep this fast.
        try:
            n_synth = len(sdf_q)
            sample_size = min(n_synth, 1000)
            nrs = NewRowSynthesis.compute(
                real_data=rdf_q,
                synthetic_data=sdf_q,
                metadata=meta_q,
                numerical_match_tolerance=0.01,
                synthetic_sample_size=sample_size,
            )
            table_result["new_row_synthesis"] = float(nrs)
        except Exception as e:
            table_result["new_row_synthesis"] = None
            table_result["new_row_synthesis_error"] = str(e)

        numeric_cols = [
            c for c, spec in single_meta["columns"].items()
            if spec.get("sdtype") == "numerical" and c in rdf_p.columns and c in sdf_p.columns
        ]
        cat_cols = [
            c for c, spec in single_meta["columns"].items()
            if spec.get("sdtype") in ("categorical", "boolean") and c in rdf_p.columns and c in sdf_p.columns
        ]

        # Nearest-neighbour distance in mixed (numeric + categorical) space.
        # Larger = synthetic sits further from real (more private).
        table_result["nn_distance"] = _nn_distance_summary(rdf_p, sdf_p, numeric_cols, cat_cols)

        # Nearest Neighbour Distance Ratio: for each synthetic row, d1/d2 where
        # d1 = distance to nearest real, d2 = distance to 2nd nearest.
        # Low NNDR means a synthetic row is uniquely close to one real record (privacy risk).
        # Higher proportion above 0.5 = better privacy (SDV convention).
        table_result["nndr"] = _nndr_summary(rdf_p, sdf_p, numeric_cols, cat_cols)

        # Fraction of synthetic rows closer to any real row than the 5th percentile
        # of real-to-real NN distances (data-driven threshold). Lower = more risky.
        table_result["privacy_risk_fraction"] = _privacy_risk_fraction(rdf_p, sdf_p, numeric_cols, cat_cols)
        table_result["score"] = _privacy_table_score(table_result)

        out[tname] = table_result

    table_scores = [r["score"] for r in out.values() if isinstance(r, dict) and r.get("score") is not None]
    out["overall_score"] = float(np.mean(table_scores)) if table_scores else None
    return out


def _privacy_table_score(table_result: Dict[str, Any]) -> Optional[float]:
    """
    Aggregate per-table privacy score (higher = more private, 0–1).

    Averages three independent signals:
    - ``new_row_synthesis``: no verbatim copies (higher = better)
    - ``nndr.proportion_above_0_5``: no memorised neighbours (higher = better)
    - ``1 - privacy_risk_fraction``: few records at singling-out risk (higher = better)
    """
    components: List[float] = []
    nrs = table_result.get("new_row_synthesis")
    if isinstance(nrs, float):
        components.append(nrs)
    nndr = table_result.get("nndr") or {}
    prop = nndr.get("proportion_above_0_5")
    if isinstance(prop, float):
        components.append(prop)
    risk = table_result.get("privacy_risk_fraction")
    if isinstance(risk, float):
        components.append(1.0 - risk)
    return float(np.mean(components)) if components else None


def _stringify_datetimes(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata: Dict[str, Any],
) -> tuple:
    """
    Return copies of (real, synth, metadata) with datetime columns coerced to
    ISO strings and reclassified as categorical.  Works around an sdmetrics
    issue where NewRowSynthesis uses ``DataFrame.query()`` with ``np.datetime64``
    literals that the query engine cannot resolve.
    """
    real_out = real.copy()
    synth_out = synth.copy()
    md_out = {"columns": dict(metadata.get("columns", {}))}
    for col, spec in list(md_out["columns"].items()):
        if col not in real_out.columns:
            continue
        sdtype = spec.get("sdtype")
        if sdtype == "datetime":
            real_out[col] = pd.to_datetime(real_out[col], errors="coerce").dt.strftime("%Y-%m-%d")
            synth_out[col] = pd.to_datetime(synth_out[col], errors="coerce").dt.strftime("%Y-%m-%d")
            md_out["columns"][col] = {"sdtype": "categorical"}
            sdtype = "categorical"
        if sdtype == "categorical":
            real_out[col] = _to_python_str_column(real_out[col], "__NULL__")
            synth_out[col] = _to_python_str_column(synth_out[col], "__NULL__")
    return real_out, synth_out, md_out


def _build_mixed_feature_matrices(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    n_real: int,
    n_synth: int,
    rng_seed: int = 0,
) -> Optional[tuple]:
    """
    Build z-scored numeric + scaled one-hot categorical feature matrices for
    nearest-neighbour distance computations.

    Categorical columns are one-hot encoded and then scaled so the entire
    categorical block contributes the same total weight as the numeric block
    (avoids either side dominating when counts differ greatly).

    Returns ``(real_matrix, synth_matrix)`` or ``None`` if no usable columns.
    """
    if not numeric_cols and not cat_cols:
        return None

    rng = np.random.default_rng(rng_seed)
    real_s = real.iloc[rng.choice(len(real), min(len(real), n_real), replace=False)]
    synth_s = synth.iloc[rng.choice(len(synth), min(len(synth), n_synth), replace=False)]

    parts_real: List[np.ndarray] = []
    parts_synth: List[np.ndarray] = []

    if numeric_cols:
        rn = real_s[numeric_cols].apply(pd.to_numeric, errors="coerce")
        sn = synth_s[numeric_cols].apply(pd.to_numeric, errors="coerce")
        meds = rn.median()
        rn, sn = rn.fillna(meds), sn.fillna(meds)
        means = rn.mean()
        stds = rn.std(ddof=0).replace(0, 1.0)
        parts_real.append(((rn - means) / stds).values.astype(float))
        parts_synth.append(((sn - means) / stds).values.astype(float))

    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder
        rc = pd.DataFrame({c: _to_python_str_column(real_s[c], "__NULL__") for c in cat_cols})
        sc = pd.DataFrame({c: _to_python_str_column(synth_s[c], "__NULL__") for c in cat_cols})
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc.fit(pd.concat([rc, sc], axis=0))
        cat_real = enc.transform(rc).astype(float)
        cat_synth = enc.transform(sc).astype(float)
        n_ohe = cat_real.shape[1]
        # Scale so categorical block has the same Frobenius weight as numeric block
        scale = (len(numeric_cols) / max(1, n_ohe)) ** 0.5 if numeric_cols else 1.0
        parts_real.append(cat_real * scale)
        parts_synth.append(cat_synth * scale)

    if not parts_real:
        return None

    return (
        np.hstack(parts_real),
        np.hstack(parts_synth),
    )


def _nn_distance_summary(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: Optional[List[str]] = None,
    n_synth_sample: int = 500,
    n_real_sample: int = 5000,
) -> Optional[Dict[str, float]]:
    """
    Median / quantile summary of the Euclidean distance from each synthetic row
    to its nearest real row, in a mixed (z-scored numeric + scaled one-hot
    categorical) feature space.

    Including categorical columns ensures that rows with matching demographics
    or clinical codes are recognised as close, not just rows with similar
    numbers.  Higher distances = synthetic data sits further from real = more
    privacy protection.
    """
    cat_cols = cat_cols or []
    if not numeric_cols and not cat_cols:
        return None

    matrices = _build_mixed_feature_matrices(
        real, synth, numeric_cols, cat_cols,
        n_real=n_real_sample, n_synth=n_synth_sample, rng_seed=0,
    )
    if matrices is None:
        return None
    r_z, s_z = matrices

    from scipy.spatial.distance import cdist
    dists = cdist(s_z, r_z, metric="euclidean")
    nn = dists.min(axis=1)
    return {
        "min": float(nn.min()),
        "p10": float(np.percentile(nn, 10)),
        "median": float(np.median(nn)),
        "p90": float(np.percentile(nn, 90)),
        "max": float(nn.max()),
    }


def _nndr_summary(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: Optional[List[str]] = None,
    n_synth_sample: int = 500,
    n_real_sample: int = 3000,
) -> Optional[Dict[str, float]]:
    """
    Nearest Neighbour Distance Ratio (NNDR) privacy metric.

    For each synthetic row, d1 = distance to nearest real row, d2 = distance
    to second nearest.  NNDR = d1/d2.

    - Low NNDR (d1 << d2): this synthetic row is unusually close to one
      specific real record — a memorisation / singling-out risk.
    - High NNDR (d1 ≈ d2): the synthetic row sits equidistant from many real
      records — no single real record can be identified.

    ``proportion_above_0_5`` (higher = better privacy) is the share of
    synthetic rows where NNDR ≥ 0.5, consistent with the SDV convention.
    ``median`` gives the central tendency.
    """
    cat_cols = cat_cols or []
    if not numeric_cols and not cat_cols:
        return None

    matrices = _build_mixed_feature_matrices(
        real, synth, numeric_cols, cat_cols,
        n_real=n_real_sample, n_synth=n_synth_sample, rng_seed=1,
    )
    if matrices is None:
        return None
    r_z, s_z = matrices

    if r_z.shape[0] < 2:
        return None

    from scipy.spatial.distance import cdist
    dists = cdist(s_z, r_z, metric="euclidean")
    sorted_dists = np.sort(dists, axis=1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]

    valid = d2 > 1e-9
    if not valid.any():
        return None

    nndr = d1[valid] / d2[valid]
    return {
        "median": float(np.median(nndr)),
        "proportion_above_0_5": float((nndr >= 0.5).mean()),
    }


def _privacy_risk_fraction(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: Optional[List[str]] = None,
    threshold_percentile: float = 5.0,
    n_sample: int = 2000,
) -> Optional[float]:
    """
    Fraction of synthetic rows that are closer to any real row than the
    ``threshold_percentile``-th percentile of real-to-real nearest-neighbour
    distances.

    Uses a data-driven threshold (anchored to actual data density) rather than
    an arbitrary distance cutoff.  Lower fraction = fewer synthetic records at
    risk of singling out a real individual.
    """
    cat_cols = cat_cols or []
    if not numeric_cols and not cat_cols:
        return None

    matrices = _build_mixed_feature_matrices(
        real, synth, numeric_cols, cat_cols,
        n_real=n_sample, n_synth=n_sample, rng_seed=2,
    )
    if matrices is None:
        return None
    r_z, s_z = matrices

    if len(r_z) < 2:
        return None

    from scipy.spatial.distance import cdist

    # Baseline: how close are real rows to each other?
    rr = cdist(r_z, r_z, metric="euclidean")
    np.fill_diagonal(rr, np.inf)
    rr_nn = rr.min(axis=1)
    threshold = float(np.percentile(rr_nn, threshold_percentile))

    # How many synthetic rows are closer to some real row than that threshold?
    sr = cdist(s_z, r_z, metric="euclidean")
    sr_nn = sr.min(axis=1)
    return float((sr_nn <= threshold).mean())


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _df_to_records(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _print_summary(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation summary ===")
    q = results.get("quality", {})
    d = results.get("diagnostic", {})
    print(f"Quality overall score    : {q.get('overall_score'):.4f}")
    if d.get("overall_score") is not None:
        print(f"Diagnostic overall score : {d['overall_score']:.4f}")
    for prop in q.get("properties", []):
        name = prop.get("Property") or prop.get("Metric") or "?"
        score = prop.get("Score")
        if score is not None:
            print(f"  · {name:<25} {score:.4f}")
    print()

    util = results.get("utility", {})
    overall_u = util.get("overall_score")
    overall_u_str = f"{overall_u:.4f}" if overall_u is not None else "n/a"
    print(f"Utility (detection — higher = more indistinguishable):  overall={overall_u_str}")
    for t, r in util.items():
        if t == "overall_score":
            continue
        v = r.get("logistic_detection")
        print(f"  · {t:<20} {v:.4f}" if v is not None else f"  · {t:<20} n/a")
    print()

    priv = results.get("privacy", {})
    overall_p = priv.get("overall_score")
    overall_p_str = f"{overall_p:.4f}" if overall_p is not None else "n/a"
    print(f"Privacy (higher = more private):  overall={overall_p_str}")
    for t, r in priv.items():
        if t == "overall_score":
            continue
        nrs = r.get("new_row_synthesis")
        nrs_str = f"{nrs:.4f}" if isinstance(nrs, float) else "n/a"
        nn = r.get("nn_distance") or {}
        nn_str = f"median NN={nn['median']:.3f}" if nn else "n/a"
        nndr = r.get("nndr") or {}
        nndr_str = f"NNDR_p50={nndr['median']:.3f}" if nndr else "n/a"
        risk = r.get("privacy_risk_fraction")
        risk_str = f"risk={risk:.3f}" if risk is not None else "n/a"
        sc = r.get("score")
        sc_str = f"score={sc:.4f}" if sc is not None else ""
        print(f"  · {t:<20} {sc_str}  new_rows={nrs_str}  {nn_str}  {nndr_str}  {risk_str}")
