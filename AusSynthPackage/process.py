"""
process(): inspect a relational dataset and emit SDC-safe metadata.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union
import base64

# SeRP branding
ROOT = Path(__file__).resolve().parent
SEPR_LOGO_PATH = ROOT / "SeRP-UK-Logo-RGB_Navy.png"
SEPR_BLUE = "#003761"
SEPR_YELLOW = "#FAEC8A"
SEPR_WHITE = "#FFFFFF"

import numpy as np
import pandas as pd

from .detect import (
    detect_dtype,
    detect_foreign_keys,
    detect_primary_key,
    is_categorical,
)
from .sdc import categorical_value_counts, continuous_bins


def process(
    tables: Dict[str, Union[str, pd.DataFrame]],
    output_path: Optional[str] = None,
    sdc_threshold: int = 10,
    n_bins: int = 10,
    max_categorical_unique: int = 20,
    primary_keys: Optional[Dict[str, Optional[str]]] = None,
    foreign_keys: Optional[Dict[str, list]] = None,
    linked_columns: Optional[Dict[str, list]] = None,
    level: int = 2,
    n_parent_context_cols: int = 3,
) -> Dict:
    """


    Build a JSON metadata document describing a relational dataset.

    Parameters
    ----------
    linked_columns : dict, optional
        Mapping of table_name -> list of lists of column names. Each inner list contains columns that should be linked (generated together).

    Parameters
    ----------
    primary_keys : dict, optional
        Mapping of table_name -> primary key column name. If provided, overrides automatic detection.
    foreign_keys : dict, optional
        Mapping of table_name -> list of foreign key dicts (with keys: column, references_table, references_column).
        If provided, overrides automatic detection.

    Parameters
    ----------
    tables : dict
        Mapping of table_name -> CSV path or pandas DataFrame.
        Example: {"patients": "patients.csv", "conditions": "conditions.csv"}
    output_path : str, optional
        If provided, the metadata JSON is also written here.
    sdc_threshold : int, default 10
        Minimum allowable count for any published category or bin.
    n_bins : int, default 10
        Initial number of equal-width bins for continuous columns
        (bins may be merged to meet the SDC threshold).
    max_categorical_unique : int, default 20
        Columns with at most this many distinct values are treated as
        categorical (for numeric columns; strings/booleans are always
        categorical).
    level : int, default 2
        Fidelity level.  level=2 (default) records per-column marginal
        distributions only.  level=3 additionally fits CART conditional
        trees that capture cross-column relationships and parent→child
        dependencies; requires scikit-learn.
    n_parent_context_cols : int, default 3
        When level=3, the number of parent-table columns to bring in as
        conditioning features for each child table.

    Returns
    -------
    dict
        The metadata document. Top-level shape:
        {
          "sdc_threshold": int,
          "tables": {
            "<table_name>": {
              "n_rows": int,
              "primary_key": str | null,
              "foreign_keys": [ {column, references_table, references_column}, ... ],
              "parent_tables": [...],     # tables this one references
              "child_tables": [...],      # tables that reference this one
              "cardinality": {            # per parent table
                  "<parent>": {"mean": float, "min": int, "max": int,
                               "p50": float, "p90": float}
              },
              "columns": {
                "<col>": {
                  "dtype": "...",
                  "completeness": float,    # fraction non-null
                  "n_null": int,
                  "is_categorical": bool,
                  "value_counts": { ... },  # if categorical
                  "bins": [ ... ],          # if continuous
                  "is_primary_key": bool,
                  "is_foreign_key": bool
                }
              }
            }
          }
        }
    """
    # --- 1. Load all tables into DataFrames ---
    loaded: Dict[str, pd.DataFrame] = {}
    for name, src in tables.items():
        if isinstance(src, pd.DataFrame):
            loaded[name] = src.copy()
        elif isinstance(src, str):
            loaded[name] = pd.read_csv(src)
        else:
            raise TypeError(
                f"Table '{name}' must be a CSV path or DataFrame, got {type(src)}"
            )

    # --- 2. Detect primary keys for every table ---
    if primary_keys is not None:
        # Use user-supplied primary keys, fill missing with None
        primary_keys = {name: primary_keys.get(name) for name in loaded}
    else:
        primary_keys = {name: detect_primary_key(df, name) for name, df in loaded.items()}

    # --- 3. Detect foreign keys ---
    if foreign_keys is not None:
        # Use user-supplied foreign keys, fill missing with empty list
        foreign_keys = {name: foreign_keys.get(name, []) for name in loaded}
    else:
        foreign_keys = detect_foreign_keys(loaded, primary_keys)

    # --- 4. Build parent/child topology ---
    parent_of: Dict[str, list] = {t: [] for t in loaded}
    children_of: Dict[str, list] = {t: [] for t in loaded}
    for child, fks in foreign_keys.items():
        for fk in fks:
            parent = fk["references_table"]
            if parent not in parent_of[child]:
                parent_of[child].append(parent)
            if child not in children_of[parent]:
                children_of[parent].append(child)

    # --- 5. Compute relational cardinality for each (parent, child) link ---
    # For each FK from child -> parent: how many child rows per parent row?
    # We attach cardinality stats to the *parent* table, keyed by child.
    cardinality: Dict[str, Dict[str, Dict]] = {t: {} for t in loaded}
    for child, fks in foreign_keys.items():
        for fk in fks:
            parent = fk["references_table"]
            parent_pk = primary_keys[parent]
            if parent_pk is None:
                continue
            child_col = fk["column"]

            # Count child rows per parent PK value (zero-fill missing parents)
            parent_keys = loaded[parent][parent_pk]
            counts = (
                loaded[child][child_col]
                .value_counts()
                .reindex(parent_keys, fill_value=0)
            )
            if len(counts) == 0:
                continue

            cardinality[parent][child] = {
                "mean": float(np.round(counts.mean(), 4)),
                "min": int(counts.min()),
                "max": int(counts.max()),
                "p50": float(np.round(counts.quantile(0.5), 4)),
                "p90": float(np.round(counts.quantile(0.9), 4)),
                "fk_column": child_col,
                # Histogram of children-per-parent so generate() can reproduce
                # the SHAPE of the distribution (zeros, skew, heavy tails),
                # not just the mean.  continuous_bins auto-detects heavy-tailed
                # distributions and switches to log-spaced edges so a long-tail
                # cardinality (most parents have few children, a few have many)
                # doesn't collapse into one giant first bin.
                "bins": continuous_bins(
                    counts.astype(float),
                    threshold=sdc_threshold,
                    n_bins=n_bins,
                ),
            }

    # --- 6. Per-column metadata with SDC ---
    metadata = {
        "sdc_threshold": sdc_threshold,
        "tables": {},
    }

    for tname, df in loaded.items():
        pk = primary_keys[tname]
        fk_cols = {fk["column"] for fk in foreign_keys[tname]}
        n_rows = len(df)

        cols_meta: Dict[str, Dict] = {}
        for col in df.columns:
            series = df[col]
            dtype = detect_dtype(series)
            original_dtype = str(series.dtype)
            n_null = int(series.isna().sum())
            completeness = float(np.round(1.0 - n_null / n_rows, 6)) if n_rows else 0.0

            col_info: Dict = {
                "dtype": dtype,
                "original_dtype": original_dtype,
                "completeness": completeness,
                "n_null": n_null,
                "is_primary_key": col == pk,
                "is_foreign_key": col in fk_cols,
            }

            # Primary keys and foreign keys are NOT distribution-modelled;
            # the generator handles them structurally.
            if col == pk or col in fk_cols:
                col_info["is_categorical"] = False
                cols_meta[col] = col_info
                continue

            categorical = is_categorical(
                series, dtype, max_unique=max_categorical_unique
            )
            col_info["is_categorical"] = categorical

            if categorical:
                col_info["value_counts"] = categorical_value_counts(
                    series, threshold=sdc_threshold
                )
            else:
                col_info["bins"] = continuous_bins(
                    series,
                    threshold=sdc_threshold,
                    n_bins=n_bins,
                    is_datetime=(dtype == "datetime"),
                )

            cols_meta[col] = col_info

        # Add linked_columns info and value pairs for this table if provided.
        # Tuple counts are SDC-filtered: tuples below the threshold are dropped
        # and (if their aggregate is itself ≥ threshold) collapsed into a single
        # __OTHER__-tuple — same rule as categorical_value_counts.
        table_linked = []
        linked_value_pairs = []
        if linked_columns is not None and tname in linked_columns:
            table_linked = linked_columns[tname]
            for group in table_linked:
                if not all(col in df.columns for col in group):
                    linked_value_pairs.append([])
                    continue
                group_df = df[group].dropna()
                value_counts = group_df.value_counts().reset_index(name="count")

                safe_pairs = []
                dropped_count = 0
                for _, row in value_counts.iterrows():
                    cnt = int(row["count"])
                    if cnt >= sdc_threshold:
                        safe_pairs.append({"values": row[group].tolist(), "count": cnt})
                    else:
                        dropped_count += cnt

                if dropped_count >= sdc_threshold:
                    safe_pairs.append({
                        "values": ["__OTHER__"] * len(group),
                        "count": dropped_count,
                    })

                linked_value_pairs.append(safe_pairs)

        # Mark non-primary columns of each linked group as derived: at generation
        # time these are looked up from the primary column's value rather than
        # generated independently.  This means semantically-redundant columns
        # (DESCRIPTION given CODE, etc.) don't waste tree depth or column-order
        # slots, and the primary remains free to act as a predictor for other
        # columns.
        for group in table_linked:
            if len(group) < 2:
                continue
            primary = group[0]
            for col in group[1:]:
                if col in cols_meta:
                    cols_meta[col]["derived_from"] = primary

        metadata["tables"][tname] = {
            "n_rows": n_rows,
            "primary_key": pk,
            "foreign_keys": foreign_keys[tname],
            "parent_tables": parent_of[tname],
            "child_tables": children_of[tname],
            "cardinality": cardinality[tname],
            "columns": cols_meta,
            "linked_columns": table_linked,
            "linked_value_pairs": linked_value_pairs,
        }

    # --- 7. Level-3: fit CART conditional trees ---
    if level >= 3:
        for tname in list(metadata["tables"].keys()):
            _attach_conditional_trees(
                tname, loaded, metadata, sdc_threshold,
                max_categorical_unique, n_parent_context_cols,
            )

    # --- 8. Optionally persist ---
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return metadata


def metadata_to_dataframe(metadata: Union[Dict, str]) -> pd.DataFrame:
    """
    Convert AusSynth JSON metadata into a human-readable DataFrame.

    Parameters
    ----------
    metadata : dict or str
        The metadata dict returned by ``process()``, or a path to the
        JSON file written by ``process(..., output_path=...)``.

    Returns
    -------
    pandas.DataFrame
        One row per column, with nested values flattened into readable text
        fields.
    """
    metadata_obj = _load_metadata_input(metadata)
    rows = []

    for table_name, table_meta in metadata_obj.get("tables", {}).items():
        columns = table_meta.get("columns", {})
        for column_name, col_meta in columns.items():
            distribution_type, distribution_text = _column_distribution_text(col_meta)
            rows.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "dtype": col_meta.get("dtype"),
                    "original_dtype": col_meta.get("original_dtype"),
                    "n_rows": table_meta.get("n_rows"),
                    "completeness": col_meta.get("completeness"),
                    "n_null": col_meta.get("n_null"),
                    "is_primary_key": col_meta.get("is_primary_key"),
                    "is_foreign_key": col_meta.get("is_foreign_key"),
                    "is_categorical": col_meta.get("is_categorical"),
                    "derived_from": col_meta.get("derived_from"),
                    "distribution_type": distribution_type,
                    "distribution": distribution_text,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "table",
            "column",
            "dtype",
            "original_dtype",
            "n_rows",
            "completeness",
            "n_null",
            "is_primary_key",
            "is_foreign_key",
            "is_categorical",
            "derived_from",
            "distribution_type",
            "distribution",
        ],
    )


def metadata_to_html(
    metadata: Union[Dict, str],
    title: str = "Metadata Report",
    output_path: Optional[str] = None,
) -> str:
    """
    Convert AusSynth JSON metadata into a collapsible HTML document.

    Parameters
    ----------
    metadata : dict or str
        The metadata dict returned by ``process()``, or a path to the
        JSON file written by ``process(..., output_path=...)``.
    title : str, default "AusSynth Metadata"
        Title shown in the rendered HTML document.
    output_path : str, optional
        If provided, the HTML document is also written to this path.

    Returns
    -------
    str
        HTML document as a string.
    """
    metadata_obj = _load_metadata_input(metadata)
    html_doc = _build_metadata_html(metadata_obj, title=title)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

    return html_doc


# ---------------------------------------------------------------------------
# Level-3 helpers: CART conditional trees
# ---------------------------------------------------------------------------

def _to_numeric_dt_aware(series: pd.Series, dtype: str) -> pd.Series:
    """
    Convert a series to a float Series for use as a tree predictor.
    Datetime columns become seconds-since-epoch (chosen over nanoseconds so the
    threshold fits comfortably in float64 with no precision loss when JSON-
    serialised).  Non-parseable values become NaN.
    """
    if dtype == "datetime":
        ts = pd.to_datetime(series, errors="coerce")
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        valid = ts.notna()
        out = pd.Series(np.full(len(series), np.nan), index=series.index, dtype=float)
        if valid.any():
            out.loc[valid] = ts[valid].astype("datetime64[ns]").astype("int64").astype(float) / 1e9
        return out
    return pd.to_numeric(series, errors="coerce")


def _load_metadata_input(metadata: Union[Dict, str]) -> Dict:
    """Load metadata from a dict or a JSON file path."""
    if isinstance(metadata, dict):
        return metadata
    with open(metadata, "r", encoding="utf-8") as f:
        return json.load(f)


def _table_summary_text(table_meta: Dict) -> str:
    parts = [f"primary_key={table_meta.get('primary_key') or 'None'}"]
    foreign_keys = table_meta.get("foreign_keys", []) or []
    if foreign_keys:
        fk_bits = [
            f"{fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}"
            for fk in foreign_keys
        ]
        parts.append("foreign_keys=" + "; ".join(fk_bits))
    linked_columns = table_meta.get("linked_columns", []) or []
    if linked_columns:
        parts.append("linked_columns=" + "; ".join([", ".join(group) for group in linked_columns]))
    if table_meta.get("cardinality"):
        parts.append(f"relationships={len(table_meta.get('cardinality', {}))}")
    return " | ".join(parts)


def _column_distribution_text(col_meta: Dict) -> tuple:
    if col_meta.get("value_counts"):
        return "categorical", _format_value_counts(col_meta.get("value_counts", {}))
    if col_meta.get("bins"):
        return "binning", _format_bins(col_meta.get("bins", []))
    return "none", "n/a"


def _format_value_counts(value_counts: Dict) -> str:
    if not value_counts:
        return "n/a"
    parts = [f"{key}: {value}" for key, value in value_counts.items()]
    return "; ".join(parts)


def _format_bins(bins: list) -> str:
    if not bins:
        return "n/a"
    parts = []
    for bin_info in bins:
        parts.append(
            f"[{_format_scalar(bin_info.get('min'))}, {_format_scalar(bin_info.get('max'))}] -> {bin_info.get('count')}"
        )
    return "; ".join(parts)


def _format_scalar(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _build_metadata_html(metadata: Dict, title: str) -> str:
        tables = metadata.get("tables", {}) or {}
        table_cards = "".join(_render_metadata_table_card(table_name, table_meta) for table_name, table_meta in tables.items())
        # embed logo if available
        def _logo_data_uri() -> Optional[str]:
            candidates = [SEPR_LOGO_PATH, ROOT / SEPR_LOGO_PATH.name]
            for p in candidates:
                try:
                    if p.exists():
                        with open(p, "rb") as f:
                            data = base64.b64encode(f.read()).decode("ascii")
                        return f"data:image/png;base64,{data}"
                except Exception:
                    continue
            return None

        logo_uri = _logo_data_uri()
        logo_img = f'<img src="{logo_uri}" alt="SeRP" style="height:64px; display:block; filter:brightness(0) invert(1);">' if logo_uri else ""

        return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: {SEPR_BLUE};
            --panel: {SEPR_WHITE};
            --panel-strong: {SEPR_WHITE};
            --ink: #0f2330;
            --muted: rgba(0,0,0,0.5);
            --accent: {SEPR_BLUE};
            --accent-2: {SEPR_YELLOW};
            --accent-soft: rgba(250,236,138,0.08);
            --border: rgba(255, 255, 255, 0.06);
            --shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
            color: var(--ink);
            background: var(--bg);
            line-height: 1.5;
        }}
        .page {{ max-width: 1220px; margin: 0 auto; padding: 28px 20px 48px; }}
        .hero {{
            background: transparent;
            color: var(--panel-strong);
            border-radius: 6px;
            padding: 18px 8px;
            box-shadow: none;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 18px;
        }}
        .hero h1 {{ margin: 0 0 4px; font-size: 1.85rem; letter-spacing: 0.01em; color: var(--accent-2); }}
        .hero p {{ margin: 0; max-width: 82ch; color: rgba(255,255,255,0.9); }}
        .summary {{
            background: var(--panel-strong);
            border: 1px solid rgba(0,0,0,0.04);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
        }}
        .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
        .metric {{
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px;
            background: linear-gradient(180deg, rgba(42,91,215,0.06), rgba(178,75,90,0.04));
        }}
        .metric .label {{ font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
        .metric .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
        .tables {{ display: grid; gap: 16px; }}
        details.table {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }}
        details.table > summary {{
            cursor: pointer;
            list-style: none;
            padding: 12px 16px;
            font-weight: 700;
            background: var(--accent);
            color: var(--accent-2);
        }}
        details.table > summary::-webkit-details-marker {{ display: none; }}
        .table-body {{ padding: 12px 14px 14px; }}
        .table-meta {{
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            margin-bottom: 14px;
            color: var(--muted);
            font-size: 0.95rem;
        }}
        details.column {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--panel-strong);
            overflow: hidden;
        }}
        details.column + details.column {{ margin-top: 10px; }}
        details.column > summary {{
            cursor: pointer;
            list-style: none;
            padding: 10px 12px;
            font-weight: 600;
            background: var(--panel-strong);
            color: var(--ink);
            position: relative;
            border-left: 4px solid rgba(0,0,0,0.04);
        }}
        details.column > summary::after {{
            content: "▾";
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            transition: transform 0.18s ease-in-out;
            color: rgba(0,0,0,0.45);
        }}
        details.column[open] > summary::after {{
            transform: translateY(-50%) rotate(180deg);
        }}
        details.column > summary::-webkit-details-marker {{ display: none; }}
        .column-body {{ padding: 12px 14px 14px; }}
        .column-grid {{
            display: grid;
            gap: 8px 14px;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin-bottom: 12px;
            font-size: 0.95rem;
        }}
        .pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: var(--accent-2);
            color: var(--accent);
            font-size: 0.84rem;
            font-weight: 700;
            margin-left: 6px;
            vertical-align: middle;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .muted {{ color: var(--muted); }}
        .subtle {{ font-size: 0.92rem; color: var(--muted); margin-bottom: 10px; }}
        .data-table {{ width: 100%; border-collapse: collapse; background: var(--panel-strong); border: 1px solid rgba(0,0,0,0.04); border-radius: 10px; overflow: hidden; }}
        .data-table th, .data-table td {{ padding: 8px 10px; border-bottom: 1px solid #edf1f6; text-align: left; vertical-align: top; }}
        .data-table th {{ background: #f5f8fc; position: sticky; top: 0; z-index: 1; }}
        .footer {{ margin-top: 18px; color: var(--panel-strong); font-size: 0.92rem; text-align: center; opacity:0.9; }}
        @media (max-width: 780px) {{
            .hero h1 {{ font-size: 1.7rem; }}
            .page {{ padding: 18px 12px 30px; }}
        }}
    </style>
</head>
<body>
    <main class="page">
        <header class="hero">
            <div style="display:flex; align-items:center; gap:14px;">
                {logo_img}
                <div>
                    <h1>{html.escape(title)}</h1>
                    <p>Expand a table to inspect its columns, then expand a column to inspect the recorded marginal distribution, completeness, key flags, and derivation metadata.</p>
                </div>
            </div>
        </header>

        <section class="summary">
            <div class="summary-grid">
                <div class="metric"><div class="label">Tables</div><div class="value">{len(tables)}</div></div>
                <div class="metric"><div class="label">Columns</div><div class="value">{sum(len((t.get('columns', {}) or {})) for t in tables.values())}</div></div>
                <div class="metric"><div class="label">SDC Threshold</div><div class="value">{html.escape(str(metadata.get('sdc_threshold', 'n/a')))}</div></div>
            </div>
        </section>

        <section class="tables">
            {table_cards if table_cards else '<p class="muted">No tables found in metadata.</p>'}
        </section>

        <div class="footer"></div>
    </main>
</body>
</html>"""


def _render_metadata_table_card(table_name: str, table_meta: Dict) -> str:
        columns = table_meta.get("columns", {}) or {}
        column_cards = "".join(_render_metadata_column_card(column_name, col_meta) for column_name, col_meta in columns.items())
        meta_bits = [
                f"Rows: {table_meta.get('n_rows', 'n/a')}",
                f"Primary key: {table_meta.get('primary_key') or 'None'}",
        ]
        if table_meta.get("foreign_keys"):
                meta_bits.append("Foreign keys: " + "; ".join(
                        f"{fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}"
                        for fk in table_meta.get("foreign_keys", [])
                ))
        if table_meta.get("child_tables"):
                meta_bits.append("Child tables: " + ", ".join(table_meta.get("child_tables", [])))
        if table_meta.get("parent_tables"):
                meta_bits.append("Parent tables: " + ", ".join(table_meta.get("parent_tables", [])))

        return f"""
        <details class="table">
            <summary>{html.escape(table_name)} <span class="pill">{len(columns)} columns</span></summary>
            <div class="table-body">
                <div class="table-meta">{''.join(f'<div>{html.escape(bit)}</div>' for bit in meta_bits)}</div>
                {column_cards if column_cards else '<p class="muted">No columns found.</p>'}
            </div>
        </details>
        """


def _render_metadata_column_card(column_name: str, col_meta: Dict) -> str:
        distribution_type, distribution_text = _column_distribution_text(col_meta)
        header_bits = []
        if col_meta.get("is_primary_key"):
                header_bits.append("Primary key")
        if col_meta.get("is_foreign_key"):
                header_bits.append("Foreign key")
        if col_meta.get("derived_from"):
                header_bits.append(f"Derived from {col_meta.get('derived_from')}")
        if col_meta.get("is_categorical"):
                header_bits.append("Categorical")
        else:
                header_bits.append("Continuous")

        summary = " | ".join(header_bits)
        metrics = [
                ("dtype", col_meta.get("dtype")),
                ("original dtype", col_meta.get("original_dtype")),
                ("completeness", col_meta.get("completeness")),
                ("nulls", col_meta.get("n_null")),
                ("distribution", distribution_type),
        ]
        metric_html = "".join(
                f'<div><strong>{html.escape(label)}:</strong> {html.escape(_format_scalar(value) if value is not None else "n/a")}</div>'
                for label, value in metrics
        )
        distribution_html = _render_distribution_html(distribution_type, distribution_text)

        return f"""
        <details class="column">
            <summary>{html.escape(column_name)} <span class="muted">{html.escape(summary)}</span></summary>
            <div class="column-body">
                <div class="column-grid">{metric_html}</div>
                {distribution_html}
            </div>
        </details>
        """


def _render_distribution_html(distribution_type: str, distribution_text: str) -> str:
        if distribution_type == "categorical":
                rows = []
                for item in distribution_text.split("; "):
                        if not item:
                                continue
                        if ": " in item:
                                category, count = item.split(": ", 1)
                        else:
                                category, count = item, ""
                        rows.append(f"<tr><td>{html.escape(category)}</td><td>{html.escape(count)}</td></tr>")
                if not rows:
                        return '<p class="subtle">No category counts available.</p>'
                return (
                        '<div class="subtle">Category value counts</div>'
                        '<table class="data-table">'
                        '<thead><tr><th>Category</th><th>Count</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody>'
                        '</table>'
                )

        if distribution_type == "binning":
                rows = []
                for item in distribution_text.split("; "):
                        if not item:
                                continue
                        if " -> " in item:
                                interval, count = item.split(" -> ", 1)
                        else:
                                interval, count = item, ""
                        rows.append(f"<tr><td>{html.escape(interval)}</td><td>{html.escape(count)}</td></tr>")
                if not rows:
                        return '<p class="subtle">No bin data available.</p>'
                return (
                        '<div class="subtle">Bin ranges</div>'
                        '<table class="data-table">'
                        '<thead><tr><th>Bin range</th><th>Count</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody>'
                        '</table>'
                )

        return f'<p class="subtle">{html.escape(distribution_text)}</p>'

def _order_columns_by_entropy(
    df: pd.DataFrame,
    content_cols: list,
    max_categorical_unique: int,
) -> list:
    """Return content_cols sorted highest-entropy first.

    Entropy is used for categorical columns; log-std for continuous.
    High-entropy columns are generated first so downstream columns have
    the most informative conditioning context.
    """
    from .detect import detect_dtype, is_categorical

    scores = []
    for col in content_cols:
        series = df[col].dropna()
        if len(series) == 0:
            scores.append((col, 0.0))
            continue
        dtype = detect_dtype(series)
        cat = is_categorical(series, dtype, max_unique=max_categorical_unique)
        if cat:
            vc = series.astype(str).value_counts(normalize=True).values
            score = float(-np.sum(vc * np.log(vc + 1e-12)))
        else:
            vals = pd.to_numeric(series, errors="coerce").dropna()
            score = float(np.log(float(vals.std()) + 1e-10)) if len(vals) > 1 else 0.0
        scores.append((col, score))

    scores.sort(key=lambda x: -x[1])
    return [col for col, _ in scores]


def _fit_and_extract_tree(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    col_meta: dict,
    sdc_threshold: int,
    max_categorical_unique: int,
) -> Optional[dict]:
    """Fit a small decision tree predicting target_col from predictor_cols.

    Returns a JSON-serialisable recursive dict describing the tree, with
    SDC-filtered distributions at every leaf.  Returns None when there is
    insufficient data to build a meaningful tree.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError:
        raise ImportError(
            "level=3 requires scikit-learn. Install with: pip install scikit-learn"
        )
    from .detect import detect_dtype, is_categorical

    # Drop rows where the target is null
    work = df[predictor_cols + [target_col]].copy().dropna(subset=[target_col])
    if len(work) < sdc_threshold * 4:
        return None
    work = work.reset_index(drop=True)

    # Encode predictor columns as numeric
    X_cols: Dict[str, np.ndarray] = {}
    feature_is_cat: Dict[str, bool] = {}
    label_encoders: Dict[str, dict] = {}  # col -> {str_val: int}
    feature_dtypes: Dict[str, str] = {}   # col -> "datetime" | "integer" | ...

    for pred_col in predictor_cols:
        series = work[pred_col]
        non_null = series.dropna()
        if len(non_null) == 0:
            X_cols[pred_col] = np.zeros(len(work), dtype=float)
            feature_is_cat[pred_col] = False
            feature_dtypes[pred_col] = "string"
            continue
        dtype = detect_dtype(non_null)
        cat = is_categorical(series, dtype, max_unique=max_categorical_unique)
        feature_is_cat[pred_col] = cat
        feature_dtypes[pred_col] = dtype
        if cat:
            str_vals = series.fillna("__NULL__").astype(str)
            cats = sorted(str_vals.unique())
            le = {c: i for i, c in enumerate(cats)}
            label_encoders[pred_col] = le
            X_cols[pred_col] = str_vals.map(le).astype(float).values
        else:
            numeric = _to_numeric_dt_aware(series, dtype)
            median = numeric.median()
            X_cols[pred_col] = numeric.fillna(median if pd.notna(median) else 0.0).values

    X = np.column_stack([X_cols[c] for c in predictor_cols])

    is_cat_target = col_meta.get("is_categorical", False)
    if is_cat_target:
        y = work[target_col].fillna("__NULL__").astype(str)
        clf = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=sdc_threshold, random_state=42
        )
    else:
        target_dtype = col_meta.get("dtype", "float")
        y = _to_numeric_dt_aware(work[target_col], target_dtype)
        valid = y.notna().values
        if valid.sum() < sdc_threshold * 4:
            return None
        X = X[valid]
        work = work[valid].reset_index(drop=True)
        y = y[valid].reset_index(drop=True)
        clf = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=sdc_threshold, random_state=42
        )

    try:
        clf.fit(X, y)
    except Exception:
        return None

    tree_dict = _extract_tree_dict(
        clf, predictor_cols, feature_is_cat, label_encoders,
        X, work, target_col, col_meta, sdc_threshold,
    )
    if tree_dict is None:
        return None
    return {"tree": tree_dict, "feature_dtypes": feature_dtypes}


def _extract_tree_dict(
    clf,
    feature_names: list,
    feature_is_cat: dict,
    label_encoders: dict,
    X_encoded: np.ndarray,
    df_train: pd.DataFrame,
    target_col: str,
    col_meta: dict,
    sdc_threshold: int,
) -> Optional[dict]:
    """Walk an sklearn tree and produce a JSON-serialisable recursive dict.

    Each leaf stores an SDC-filtered conditional distribution in the same
    value_counts / bins format used by the existing per-column marginals.
    """
    from .sdc import categorical_value_counts, continuous_bins

    sk_tree = clf.tree_
    dtype = col_meta.get("dtype", "float")
    is_cat_target = col_meta.get("is_categorical", False)

    global_bins = col_meta.get("bins", [])

    def leaf_dist(row_idx: np.ndarray) -> Optional[dict]:
        subset = df_train.iloc[row_idx][target_col]
        if is_cat_target:
            vc = categorical_value_counts(subset, sdc_threshold)
            return {"value_counts": vc} if vc else None
        else:
            # Reuse the global bin edges so all leaf distributions compose back
            # to the global marginal rather than drifting to leaf-local ranges.
            # Pass sdc_threshold so per-bin leaf counts are also SDC-filtered.
            if global_bins:
                bins = _leaf_bins_from_global(
                    subset, global_bins, dtype, sdc_threshold=sdc_threshold,
                )
            else:
                bins = continuous_bins(
                    subset, threshold=sdc_threshold, is_datetime=(dtype == "datetime")
                )
            return {"bins": bins} if bins else None

    def recurse(node_id: int, row_idx: np.ndarray) -> Optional[dict]:
        if len(row_idx) == 0:
            return None
        if sk_tree.children_left[node_id] == -1:  # leaf
            dist = leaf_dist(row_idx)
            return {"type": "leaf", **dist} if dist else None

        feat_idx = int(sk_tree.feature[node_id])
        feat_name = feature_names[feat_idx]
        threshold = float(sk_tree.threshold[node_id])

        feat_col = X_encoded[row_idx, feat_idx]
        left_mask = feat_col <= threshold
        left_idx = row_idx[left_mask]
        right_idx = row_idx[~left_mask]

        if feature_is_cat.get(feat_name, False):
            le = label_encoders.get(feat_name, {})
            cats_left = sorted(
                c for c, enc in le.items()
                if enc <= threshold and c != "__NULL__"
            )
            node: dict = {
                "type": "split",
                "feature": feat_name,
                "is_categorical": True,
                "categories_left": cats_left,
            }
        else:
            node = {
                "type": "split",
                "feature": feat_name,
                "is_categorical": False,
                "threshold": round(threshold, 6),
            }

        left_child = recurse(sk_tree.children_left[node_id], left_idx)
        right_child = recurse(sk_tree.children_right[node_id], right_idx)

        # If either child failed SDC, collapse to a leaf over the whole partition
        if left_child is None or right_child is None:
            dist = leaf_dist(row_idx)
            return {"type": "leaf", **dist} if dist else None

        node["left"] = left_child
        node["right"] = right_child
        return node

    return recurse(0, np.arange(len(df_train)))


def _leaf_bins_from_global(
    series: pd.Series,
    global_bins: list,
    dtype: str,
    sdc_threshold: Optional[int] = None,
) -> list:
    """
    Count series values within the global bin edges rather than recomputing
    bin edges from scratch.  This keeps all leaf distributions on the same
    axis as the global marginal so they compose correctly during generation.

    When ``sdc_threshold`` is provided, per-bin counts below that threshold
    are suppressed entirely so the published tree metadata does not expose
    low-frequency values from a leaf distribution.  Returns [] when nothing
    survives, which lets ``_extract_tree_dict`` collapse the node up to its
    parent.
    """
    is_datetime = dtype == "datetime"

    if is_datetime:
        vals = pd.to_datetime(series, errors="coerce").dropna()
        if vals.empty:
            return []
        if hasattr(vals.dt, "tz") and vals.dt.tz is not None:
            vals = vals.dt.tz_localize(None)
        vals = vals.astype("datetime64[ns]").astype("int64").values
        edges_lo = [pd.Timestamp(b["min"]).value for b in global_bins]
        edges_hi = [pd.Timestamp(b["max"]).value for b in global_bins]
    else:
        vals = pd.to_numeric(series, errors="coerce").dropna().values
        if len(vals) == 0:
            return []
        edges_lo = [float(b["min"]) for b in global_bins]
        edges_hi = [float(b["max"]) for b in global_bins]

    result = []
    n = len(global_bins)
    for i, b in enumerate(global_bins):
        lo, hi = edges_lo[i], edges_hi[i]
        if lo == hi:
            # Point bin (zero-inflation peel-off, or a single-value column).
            # The standard half-open histogram interval [lo, hi) would never
            # match anything, so count exact equals instead.
            count = int(np.sum(vals == lo))
        elif i < n - 1:
            count = int(np.sum((vals >= lo) & (vals < hi)))
        else:
            count = int(np.sum((vals >= lo) & (vals <= hi)))
        # Per-bin SDC: suppress low-frequency bins rather than publishing
        # them with a small or zero count.
        if sdc_threshold is not None and count < sdc_threshold:
            continue
        result.append({"min": b["min"], "max": b["max"], "count": count})

    # Return [] when nothing survives (so _extract_tree_dict can collapse
    # this subtree up to its parent and we don't publish an unusable leaf).
    return result if any(b["count"] > 0 for b in result) else []


def _attach_conditional_trees(
    tname: str,
    loaded: dict,
    metadata: dict,
    sdc_threshold: int,
    max_categorical_unique: int,
    n_parent_context_cols: int,
) -> None:
    """Fit and attach CART conditional trees to one table's metadata entry."""
    try:
        from sklearn.tree import DecisionTreeClassifier  # availability check
    except ImportError:
        raise ImportError(
            "level=3 requires scikit-learn. Install with: pip install scikit-learn"
        )

    tmeta = metadata["tables"][tname]
    df = loaded[tname].copy()
    pk = tmeta["primary_key"]
    fk_cols = {fk["column"] for fk in tmeta["foreign_keys"]}
    # Derived columns (secondary members of linked groups) are not modelled by
    # trees — at generation time they are looked up from their primary.
    derived_cols = {
        c for c, cmeta in tmeta["columns"].items() if cmeta.get("derived_from")
    }
    content_cols = [
        c for c in df.columns
        if c != pk and c not in fk_cols and c not in derived_cols
    ]

    # High-cardinality string columns (for example first/last names or
    # addresses) are not useful in conditional trees and can expose sensitive
    # low-frequency content in the published level-3 metadata. Exclude them
    # from both target and predictor roles.
    tree_safe_cols = [
        c for c in content_cols
        if not _is_tree_sensitive_column(df[c], tmeta["columns"].get(c, {}), max_categorical_unique)
    ]

    if len(tree_safe_cols) < 2:
        tmeta["column_order"] = tree_safe_cols
        tmeta["conditional_trees"] = {}
        tmeta["parent_context_columns"] = []
        return

    # Join parent attributes as "parent__<col>" context columns
    parent_context_cols: list = []
    for fk in tmeta["foreign_keys"]:
        parent_name = fk["references_table"]
        if parent_name not in loaded:
            continue
        parent_df = loaded[parent_name]
        parent_pk = metadata["tables"][parent_name]["primary_key"]
        parent_fk_set = {
            f["column"] for f in metadata["tables"][parent_name].get("foreign_keys", [])
        }
        parent_attrs = [
            c for c in parent_df.columns
            if c != parent_pk and c not in parent_fk_set
        ][:n_parent_context_cols]

        parent_attrs = [
            c for c in parent_attrs
            if not _is_tree_sensitive_column(
                parent_df[c],
                metadata["tables"][parent_name]["columns"].get(c, {}),
                max_categorical_unique,
            )
        ]

        if not parent_attrs:
            continue

        rename_map = {c: f"parent__{c}" for c in parent_attrs}
        parent_subset = (
            parent_df[[parent_pk] + parent_attrs]
            .rename(columns=rename_map)
        )
        df = df.merge(
            parent_subset,
            left_on=fk["column"],
            right_on=parent_pk,
            how="left",
        ).drop(columns=[parent_pk], errors="ignore")

        for c in parent_attrs:
            prefixed = f"parent__{c}"
            if prefixed in df.columns:
                parent_context_cols.append(prefixed)

    # Determine generation order by entropy
    col_order = _order_columns_by_entropy(df, tree_safe_cols, max_categorical_unique)

    # Fit a conditional tree for each content column.  The first column has no
    # preceding within-table predictors, but for child tables the parent context
    # is already known at generation time, so we still fit a tree using just
    # those parent columns.  For parent tables (no parent_context_cols), the
    # first column genuinely has no predictors and falls back to its marginal.
    conditional_trees: dict = {}
    for i, col in enumerate(col_order):
        if i == 0:
            predictors = [c for c in parent_context_cols if c in df.columns]
        else:
            predictors = [c for c in col_order[:i] + parent_context_cols if c in df.columns]

        if not predictors:
            continue

        col_meta_entry = tmeta["columns"].get(col)
        if col_meta_entry is None:
            continue

        fit_result = _fit_and_extract_tree(
            df, col, predictors, col_meta_entry, sdc_threshold, max_categorical_unique
        )
        if fit_result is not None:
            conditional_trees[col] = {
                "predictor_columns": predictors,
                "predictor_dtypes": fit_result["feature_dtypes"],
                "tree": fit_result["tree"],
            }

    tmeta["column_order"] = col_order
    tmeta["conditional_trees"] = conditional_trees
    tmeta["parent_context_columns"] = parent_context_cols


def _is_tree_sensitive_column(series: pd.Series, col_meta: Dict, max_categorical_unique: int) -> bool:
    """Heuristically exclude free-text / high-cardinality string columns from trees."""
    dtype = col_meta.get("dtype") or detect_dtype(series)
    if dtype != "string":
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    n_unique = int(non_null.astype(str).nunique())
    n_total = int(len(non_null))
    unique_ratio = n_unique / max(1, n_total)

    # String columns with many distinct values relative to table size tend to
    # behave like names, addresses, or other free-text identifiers. These are
    # intentionally not modelled in level-3 trees.
    return n_unique > max(max_categorical_unique * 3, 50) and unique_ratio > 0.05
