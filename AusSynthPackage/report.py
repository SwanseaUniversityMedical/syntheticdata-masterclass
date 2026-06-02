"""Generate an automated HTML report for AusSynth evaluations.

The report combines the four evaluation sections returned by
``AusSynthPackage.evaluate`` with a compact visual audit of the data:

- 2 categorical distribution comparisons
- 2 continuous distribution comparisons
- 2 boxplots with one categorical and one continuous variable each
- 2 continuous scatter plots
- 1 cardinality plot for a parent/child relationship
- a relationship summary showing inter-table links

The module is intentionally usable both as a library and as a script.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AusSynthPackage import evaluate  # noqa: E402

# SeRP branding
SEPR_LOGO_PATH = ROOT / "SeRP-UK-Logo-RGB_Navy.png"
SEPR_BLUE = "#003761"
SEPR_YELLOW = "#FAEC8A"
SEPR_YELLOW_DARK = "#C9A22A"
SEPR_WHITE = "#FFFFFF"


TableSource = Union[str, pd.DataFrame]
TableMap = Dict[str, pd.DataFrame]


def generate_html_report(
    real_tables: Union[Mapping[str, TableSource], str],
    synthetic_tables: Union[Mapping[str, TableSource], str],
    metadata: Union[Dict[str, Any], str],
    output_path: str = "report.html",
    title: str = "Synthetic Data Report",
    seed: int = 0,
    exclude_columns: Optional[Dict[str, List[str]]] = None,
    plot_spec: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    """Generate and save an HTML report.

    Parameters
    ----------
    real_tables, synthetic_tables
        Either ``{table_name: DataFrame/CSV path}``, a directory containing
        ``<table>.csv`` files, or a JSON file containing such a mapping.
    metadata
        The AusSynth metadata dict or a path to the JSON metadata produced by
        ``process()``.
    output_path
        Destination HTML file.
    title
        Page title shown in the report header.
    seed
        Seed used for random visual selection.
    exclude_columns
        Optional ``{table: [columns...]}`` exclusion map passed through to
        ``evaluate``.
    plot_spec
        Optional manual plot selection map. Supported keys are
        ``categorical_distributions``, ``continuous_distributions``,
        ``boxplots``, ``scatter_plots``, ``correlation_tables``, and
        ``cardinality_relationships``. When a key is provided, the report uses
        those selections instead of random picks for that graph family.
    verbose
        Print a short status line while generating the report.
    """
    rng = np.random.default_rng(seed)
    real = _load_table_map(real_tables)
    synth = _load_table_map(synthetic_tables)
    metadata_obj = _load_metadata(metadata)

    if verbose:
        print("[report] running evaluations...")
    results = evaluate(
        real_tables=real,
        synthetic_tables=synth,
        metadata=metadata_obj,
        exclude_columns=exclude_columns,
        output_path=None,
        verbose=False,
    )

    if verbose:
        print("[report] building visualisations...")
    figures = _build_figures(real, synth, metadata_obj, rng, plot_spec=plot_spec)
    html_doc = _build_html_document(title, results, figures, real, synth, metadata_obj)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")

    if verbose:
        print(f"[report] wrote {out_path}")
    return str(out_path)


def _load_metadata(metadata: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    with open(metadata, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_table_map(source: Union[Mapping[str, TableSource], str]) -> TableMap:
    if isinstance(source, Mapping):
        return {name: _load_table_value(value) for name, value in source.items()}

    path = Path(source)
    if path.is_dir():
        tables: TableMap = {}
        for csv_path in sorted(path.glob("*.csv")):
            tables[csv_path.stem] = pd.read_csv(csv_path)
        if not tables:
            raise ValueError(f"No CSV files found in directory: {source}")
        return tables

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise TypeError(f"JSON table map must be an object: {source}")
        return {name: _load_table_value(value) for name, value in mapping.items()}

    raise TypeError(
        "Table source must be a mapping, a directory of CSV files, or a JSON mapping file"
    )


def _load_table_value(value: TableSource) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, str):
        return pd.read_csv(value)
    raise TypeError(f"Unsupported table source type: {type(value)}")


def _build_figures(
    real: TableMap,
    synth: TableMap,
    metadata: Dict[str, Any],
    rng: np.random.Generator,
    plot_spec: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    sns.set_theme(style="whitegrid", context="talk", palette=[SEPR_BLUE, SEPR_YELLOW])
    figures: List[Dict[str, Any]] = []
    available = _column_catalog(real, metadata)
    spec = _normalize_plot_spec(plot_spec)

    categorical_picks = spec["categorical_distributions"] or _pick_columns(available, kind="categorical", count=2, rng=rng)
    for table, column in categorical_picks:
        figures.append(_plot_categorical_distribution(real, synth, table, column))

    continuous_picks = spec["continuous_distributions"] or _pick_columns(available, kind="continuous", count=2, rng=rng)
    for table, column in continuous_picks:
        figures.append(_plot_continuous_distribution(real, synth, table, column))

    boxplot_picks = spec["boxplots"] or _pick_boxplot_pairs(available, count=2, rng=rng)
    for table, pair in boxplot_picks:
        figures.append(_plot_boxplot_distribution(real, synth, table, pair[0], pair[1]))

    scatter_picks = spec["scatter_plots"] or _pick_scatter_pairs(available, count=2, rng=rng)
    for table, pair in scatter_picks:
        figures.append(_plot_continuous_scatter(real, synth, table, pair[0], pair[1]))

    correlation_tables = spec["correlation_tables"] or _tables_with_numeric_correlations(real, synth)
    for table in correlation_tables:
        corr_fig = _plot_correlation_heatmap(real, synth, table)
        if corr_fig is not None:
            figures.append(corr_fig)

    if spec["cardinality_relationships"]:
        for parent, child in spec["cardinality_relationships"]:
            cardinality = _plot_cardinality_relationship(real, synth, metadata, rng, relationship_override=(parent, child))
            if cardinality is not None:
                figures.append(cardinality)
    else:
        cardinality = _plot_cardinality_relationship(real, synth, metadata, rng)
        if cardinality is not None:
            figures.append(cardinality)

    return figures


def _normalize_plot_spec(plot_spec: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
    normalized: Dict[str, List[Any]] = {
        "categorical_distributions": [],
        "continuous_distributions": [],
        "boxplots": [],
        "scatter_plots": [],
        "correlation_tables": [],
        "cardinality_relationships": [],
    }
    if not plot_spec:
        return normalized

    def _as_table_column_list(items: Any) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for item in items or []:
            if isinstance(item, dict):
                table = item.get("table")
                column = item.get("column")
            else:
                table, column = item
            if table is None or column is None:
                raise ValueError("Plot spec entries must include table and column names")
            pairs.append((str(table), str(column)))
        return pairs

    def _as_boxplot_list(items: Any) -> List[Tuple[str, Tuple[str, str]]]:
        pairs: List[Tuple[str, Tuple[str, str]]] = []
        for item in items or []:
            if isinstance(item, dict):
                table = item.get("table")
                category = item.get("category") or item.get("category_col")
                value = item.get("value") or item.get("value_col")
            else:
                table, category, value = item
            if table is None or category is None or value is None:
                raise ValueError("Boxplot spec entries must include table, category, and value names")
            pairs.append((str(table), (str(category), str(value))))
        return pairs

    def _as_scatter_list(items: Any) -> List[Tuple[str, Tuple[str, str]]]:
        pairs: List[Tuple[str, Tuple[str, str]]] = []
        for item in items or []:
            if isinstance(item, dict):
                table = item.get("table")
                x_col = item.get("x") or item.get("x_col")
                y_col = item.get("y") or item.get("y_col")
            else:
                table, x_col, y_col = item
            if table is None or x_col is None or y_col is None:
                raise ValueError("Scatter spec entries must include table, x, and y names")
            pairs.append((str(table), (str(x_col), str(y_col))))
        return pairs

    def _as_table_list(items: Any) -> List[str]:
        return [str(item) for item in (items or [])]

    def _as_relationship_list(items: Any) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for item in items or []:
            if isinstance(item, dict):
                parent = item.get("parent") or item.get("parent_table")
                child = item.get("child") or item.get("child_table")
            else:
                parent, child = item
            if parent is None or child is None:
                raise ValueError("Cardinality spec entries must include parent and child table names")
            pairs.append((str(parent), str(child)))
        return pairs

    alias_map = {
        "categorical": "categorical_distributions",
        "categorical_distributions": "categorical_distributions",
        "continuous": "continuous_distributions",
        "continuous_distributions": "continuous_distributions",
        "boxplots": "boxplots",
        "boxplot_pairs": "boxplots",
        "scatter": "scatter_plots",
        "scatter_plots": "scatter_plots",
        "correlations": "correlation_tables",
        "correlation_tables": "correlation_tables",
        "cardinality": "cardinality_relationships",
        "cardinality_relationships": "cardinality_relationships",
    }

    for key, value in plot_spec.items():
        normalized_key = alias_map.get(key)
        if normalized_key is None:
            continue
        if normalized_key == "categorical_distributions":
            normalized[normalized_key] = _as_table_column_list(value)
        elif normalized_key == "continuous_distributions":
            normalized[normalized_key] = _as_table_column_list(value)
        elif normalized_key == "boxplots":
            normalized[normalized_key] = _as_boxplot_list(value)
        elif normalized_key == "scatter_plots":
            normalized[normalized_key] = _as_scatter_list(value)
        elif normalized_key == "correlation_tables":
            normalized[normalized_key] = _as_table_list(value)
        elif normalized_key == "cardinality_relationships":
            normalized[normalized_key] = _as_relationship_list(value)

    return normalized


def _column_catalog(
    tables: TableMap,
    metadata: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    catalog: Dict[str, Dict[str, List[str]]] = {}
    meta_tables = metadata.get("tables", {})
    for tname, df in tables.items():
        tmeta = meta_tables.get(tname, {})
        cols = tmeta.get("columns", {})
        categorical: List[str] = []
        continuous: List[str] = []
        discrete: List[str] = []
        for col in df.columns:
            cmeta = cols.get(col, {})
            if cmeta.get("derived_from"):
                continue
            if cmeta.get("is_primary_key") or cmeta.get("is_foreign_key"):
                continue
            if cmeta.get("is_categorical", False):
                if cmeta.get("dtype") == "string":
                    if _is_low_cardinality_text(df[col], max_unique=20):
                        categorical.append(col)
                elif _is_plot_ready_series(df[col], allow_text=False):
                    categorical.append(col)
                    if _is_discrete_like(cmeta, df[col]):
                        discrete.append(col)
            else:
                if _is_plot_ready_series(df[col], allow_text=False):
                    continuous.append(col)
                    if _is_discrete_like(cmeta, df[col]):
                        discrete.append(col)
        catalog[tname] = {
            "categorical": categorical,
            "continuous": continuous,
            "discrete": list(dict.fromkeys(discrete)),
        }
    return catalog


def _is_plot_ready_series(series: pd.Series, allow_text: bool) -> bool:
    non_null = series.dropna()
    if len(non_null) < 2:
        return False
    if allow_text:
        return True
    numeric = pd.to_numeric(non_null, errors="coerce")
    return numeric.notna().sum() >= 2


def _is_low_cardinality_text(series: pd.Series, max_unique: int) -> bool:
    non_null = series.dropna().astype(str)
    if len(non_null) < 2:
        return False
    return non_null.nunique(dropna=True) <= max_unique


def _is_discrete_like(cmeta: Dict[str, Any], series: pd.Series) -> bool:
    dtype = cmeta.get("dtype")
    if cmeta.get("is_categorical", False):
        return True
    if dtype in ("integer", "boolean"):
        return True
    if dtype == "datetime":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    nunique = non_null.nunique(dropna=True)
    return nunique <= 12


def _pick_columns(
    catalog: Dict[str, Dict[str, List[str]]],
    kind: str,
    count: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    tables = list(catalog.keys())
    rng.shuffle(tables)
    selected: List[Tuple[str, str]] = []
    used: set = set()

    for table in tables:
        candidates = [c for c in catalog[table].get(kind, []) if (table, c) not in used]
        if not candidates:
            continue
        chosen = str(rng.choice(candidates))
        selected.append((table, chosen))
        used.add((table, chosen))
        if len(selected) >= count:
            break

    if len(selected) < count:
        pool = [
            (table, column)
            for table, kinds in catalog.items()
            for column in kinds.get(kind, [])
            if (table, column) not in used
        ]
        rng.shuffle(pool)
        for item in pool:
            selected.append(item)
            if len(selected) >= count:
                break

    return selected[:count]


def _pick_boxplot_pairs(
    catalog: Dict[str, Dict[str, List[str]]],
    count: int,
    rng: np.random.Generator,
) -> List[Tuple[str, Tuple[str, str]]]:
    tables = [t for t, kinds in catalog.items() if kinds.get("categorical") and kinds.get("continuous")]
    rng.shuffle(tables)
    pairs: List[Tuple[str, Tuple[str, str]]] = []

    for table in tables:
        categories = catalog[table].get("categorical", [])[:]
        values = catalog[table].get("continuous", [])[:]
        rng.shuffle(categories)
        rng.shuffle(values)
        pairs.append((table, (categories[0], values[0])))
        if len(pairs) >= count:
            break

    if len(pairs) < count:
        for table, kinds in catalog.items():
            categories = kinds.get("categorical", [])
            values = kinds.get("continuous", [])
            if not categories or not values:
                continue
            pair = (categories[0], values[0])
            if (table, pair) not in pairs:
                pairs.append((table, pair))
            if len(pairs) >= count:
                break

    return pairs[:count]


def _pick_scatter_pairs(
    catalog: Dict[str, Dict[str, List[str]]],
    count: int,
    rng: np.random.Generator,
) -> List[Tuple[str, Tuple[str, str]]]:
    tables = [t for t, kinds in catalog.items() if len(kinds.get("continuous", [])) >= 2]
    rng.shuffle(tables)
    pairs: List[Tuple[str, Tuple[str, str]]] = []

    for table in tables:
        columns = catalog[table].get("continuous", [])[:]
        rng.shuffle(columns)
        pairs.append((table, (columns[0], columns[1])))
        if len(pairs) >= count:
            break

    if len(pairs) < count:
        for table, kinds in catalog.items():
            columns = kinds.get("continuous", [])
            if len(columns) < 2:
                continue
            pair = (columns[0], columns[1])
            if (table, pair) not in pairs:
                pairs.append((table, pair))
            if len(pairs) >= count:
                break

    return pairs[:count]


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _plot_continuous_scatter(
    real: TableMap,
    synth: TableMap,
    table: str,
    x_col: str,
    y_col: str,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    real_x = pd.to_numeric(real[table][x_col], errors="coerce")
    real_y = pd.to_numeric(real[table][y_col], errors="coerce")
    synth_x = pd.to_numeric(synth[table][x_col], errors="coerce")
    synth_y = pd.to_numeric(synth[table][y_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _sns_scatter_sample(ax, real_x, real_y, label="Real", color=SEPR_BLUE)
    _sns_scatter_sample(ax, synth_x, synth_y, label="Synthetic", color=SEPR_YELLOW_DARK)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter plot: {table}.{x_col} vs {table}.{y_col}")
    ax.legend()
    sns.despine(ax=ax)

    return {
        "title": f"Scatter plot: {table}.{x_col} vs {table}.{y_col}",
        "caption": f"Real and synthetic scatter comparison for {table}.{x_col} and {table}.{y_col}",
        "image": _figure_to_data_uri(fig),
    }


def _sns_scatter_sample(ax, x: pd.Series, y: pd.Series, label: str, color: str) -> None:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return
    if len(x) > 1500:
        idx = np.random.default_rng(0).choice(len(x), size=1500, replace=False)
        x = x.iloc[idx]
        y = y.iloc[idx]
    sns.scatterplot(x=x, y=y, ax=ax, s=18, alpha=0.38, color=color, label=label, edgecolor=None)


def _logo_data_uri() -> Optional[str]:
    candidates = [SEPR_LOGO_PATH, ROOT / "AusSynthPackage" / SEPR_LOGO_PATH.name]
    for p in candidates:
        try:
            if p.exists():
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                return f"data:image/png;base64,{data}"
        except Exception:
            continue
    return None


def _plot_categorical_distribution(
    real: TableMap,
    synth: TableMap,
    table: str,
    column: str,
) -> Dict[str, Any]:
    r = real[table][column].astype(object)
    s = synth[table][column].astype(object)
    categories = _top_categories(r, s, max_categories=10)
    r_counts = _count_categories(r, categories)
    s_counts = _count_categories(s, categories)

    import matplotlib.pyplot as plt

    plot_df = pd.DataFrame({
        "category": categories,
        "Real": r_counts,
        "Synthetic": s_counts,
    }).melt(id_vars="category", var_name="dataset", value_name="count")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.barplot(
        data=plot_df,
        x="category",
        y="count",
        hue="dataset",
        palette=[SEPR_BLUE, SEPR_YELLOW],
        ax=ax,
        errorbar=None,
    )
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Categorical distribution: {table}.{column}")
    ax.legend()
    sns.despine(ax=ax)

    return {
        "title": f"Categorical distribution: {table}.{column}",
        "caption": f"Real vs synthetic frequency comparison for {table}.{column}",
        "image": _figure_to_data_uri(fig),
    }


def _top_categories(real: pd.Series, synth: pd.Series, max_categories: int = 10) -> List[str]:
    values = pd.concat([real, synth], axis=0).astype(object)
    values = values.where(values.notna(), "__NULL__").astype(str)
    counts = values.value_counts()
    categories = [c for c in counts.head(max_categories - 1).index.tolist() if c != "__OTHER__"]
    if len(counts) > len(categories):
        categories.append("__OTHER__")
    return categories


def _count_categories(series: pd.Series, categories: Sequence[str]) -> np.ndarray:
    values = series.astype(object)
    values = values.where(values.notna(), "__NULL__").astype(str)
    counts = []
    other = 0
    for category in categories:
        if category == "__OTHER__":
            continue
        counts.append(int((values == category).sum()))
    if "__OTHER__" in categories:
        known = set(c for c in categories if c != "__OTHER__")
        other = int((~values.isin(known)).sum())
        counts.append(other)
    return np.asarray(counts, dtype=int)


def _plot_continuous_distribution(
    real: TableMap,
    synth: TableMap,
    table: str,
    column: str,
) -> Dict[str, Any]:
    r = pd.to_numeric(real[table][column], errors="coerce").dropna()
    s = pd.to_numeric(synth[table][column], errors="coerce").dropna()
    bins = _shared_bins(r, s)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.histplot(r, bins=bins, stat="density", element="step", fill=True, alpha=0.22, color=SEPR_BLUE, label="Real", ax=ax)
    sns.histplot(s, bins=bins, stat="density", element="step", fill=True, alpha=0.22, color=SEPR_YELLOW, label="Synthetic", ax=ax)
    ax.set_title(f"Continuous distribution: {table}.{column}")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)

    return {
        "title": f"Continuous distribution: {table}.{column}",
        "caption": f"Overlaid histograms for {table}.{column}",
        "image": _figure_to_data_uri(fig),
    }


def _shared_bins(real: pd.Series, synth: pd.Series, max_bins: int = 30) -> int:
    combined = pd.concat([real, synth], axis=0)
    if combined.empty:
        return 10
    return max(5, min(max_bins, int(np.sqrt(len(combined)))))


def _plot_boxplot_distribution(
    real: TableMap,
    synth: TableMap,
    table: str,
    category_col: str,
    value_col: str,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    real_df = pd.DataFrame({
        "dataset": "Real",
        "category": real[table][category_col].astype(object),
        "value": pd.to_numeric(real[table][value_col], errors="coerce"),
    })
    synth_df = pd.DataFrame({
        "dataset": "Synthetic",
        "category": synth[table][category_col].astype(object),
        "value": pd.to_numeric(synth[table][value_col], errors="coerce"),
    })
    plot_df = pd.concat([real_df, synth_df], ignore_index=True)
    plot_df = plot_df.dropna(subset=["category", "value"])
    if plot_df.empty:
        raise ValueError(f"No plot-ready data for {table}.{category_col} vs {value_col}")

    categories = _top_categories(plot_df["category"], plot_df["category"], max_categories=10)
    plot_df["category"] = plot_df["category"].astype(object).where(plot_df["category"].astype(object).isin(categories), "__OTHER__")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=plot_df,
        x="category",
        y="value",
        hue="dataset",
        order=categories,
        palette=[SEPR_BLUE, SEPR_YELLOW],
        ax=ax,
        showfliers=False,
    )
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"Boxplot: {table}.{value_col} by {category_col}")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    sns.despine(ax=ax)

    return {
        "title": f"Boxplot: {table}.{value_col} by {category_col}",
        "caption": f"Real and synthetic boxplots for {table}.{value_col} grouped by {category_col}",
        "image": _figure_to_data_uri(fig),
    }


def _tables_with_numeric_correlations(real: TableMap, synth: TableMap) -> List[str]:
    tables: List[str] = []
    for tname in real.keys():
        if tname not in synth:
            continue
        real_numeric = real[tname].select_dtypes(include=[np.number])
        synth_numeric = synth[tname].select_dtypes(include=[np.number])
        common_cols = [c for c in real_numeric.columns if c in synth_numeric.columns]
        if len(common_cols) >= 2:
            tables.append(tname)
    return tables


def _plot_correlation_heatmap(
    real: TableMap,
    synth: TableMap,
    table: str,
) -> Optional[Dict[str, Any]]:
    real_numeric = real[table].select_dtypes(include=[np.number])
    synth_numeric = synth[table].select_dtypes(include=[np.number])
    columns = [c for c in real_numeric.columns if c in synth_numeric.columns]
    if len(columns) < 2:
        return None

    real_corr = real_numeric[columns].corr()
    synth_corr = synth_numeric[columns].corr()
    combined = np.nan_to_num(np.concatenate([real_corr.values.ravel(), synth_corr.values.ravel()]), nan=0.0)
    vmax = float(np.max(np.abs(combined))) if combined.size else 1.0
    vmax = max(vmax, 0.3)

    import matplotlib.pyplot as plt

    width = max(8.5, 0.42 * len(columns) * 2)
    height = max(6.0, 0.34 * len(columns) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
    heatmap_opts = dict(cmap="coolwarm", vmin=-vmax, vmax=vmax, center=0.0, square=True, linewidths=0.3, linecolor="white")

    sns.heatmap(real_corr, ax=axes[0], cbar=False, xticklabels=False, yticklabels=False, **heatmap_opts)
    axes[0].set_title(f"Real: {table}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", bottom=False, top=False)
    axes[0].tick_params(axis="y", left=False, right=False)

    sns.heatmap(synth_corr, ax=axes[1], cbar=True, xticklabels=False, yticklabels=False, **heatmap_opts)
    axes[1].set_title(f"Synthetic: {table}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", bottom=False, top=False)
    axes[1].tick_params(axis="y", left=False, right=False)

    return {
        "title": f"Correlation audit: {table}",
        "caption": f"Side-by-side correlation matrices for numeric columns in {table}",
        "image": _figure_to_data_uri(fig),
    }


def _plot_cardinality_relationship(
    real: TableMap,
    synth: TableMap,
    metadata: Dict[str, Any],
    rng: np.random.Generator,
    relationship_override: Optional[Tuple[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    if relationship_override is None:
        relationship = _choose_relationship(metadata, rng)
    else:
        parent, child = relationship_override
        relationship = _resolve_relationship(metadata, parent, child)

    if relationship is None:
        return None

    parent, child, fk_column, pk_column = relationship
    real_counts = real[child][fk_column].value_counts().sort_index()
    synth_counts = synth[child][fk_column].value_counts().sort_index()
    parent_keys = metadata["tables"][parent].get("n_rows", len(real[parent]))
    real_counts = real[parent][pk_column].map(real_counts).fillna(0)
    synth_counts = synth[parent][pk_column].map(synth_counts).fillna(0)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = _shared_bins(real_counts, synth_counts, max_bins=25)
    sns.histplot(real_counts, bins=bins, stat="density", element="step", fill=True, alpha=0.24, color=SEPR_BLUE, label="Real", ax=ax)
    sns.histplot(synth_counts, bins=bins, stat="density", element="step", fill=True, alpha=0.24, color=SEPR_YELLOW, label="Synthetic", ax=ax)
    ax.set_title(f"Child cardinality per parent: {parent} -> {child}")
    ax.set_xlabel(f"{child} rows per {parent} parent")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)

    summary = metadata.get("tables", {}).get(parent, {}).get("cardinality", {}).get(child, {})
    subtitle = (
        f"Relationship: {parent}.{pk_column} -> {child}.{fk_column} | "
        f"mean={summary.get('mean', 'n/a')} min={summary.get('min', 'n/a')} max={summary.get('max', 'n/a')}"
    )
    return {
        "title": f"Cardinality: {parent} -> {child}",
        "caption": subtitle,
        "image": _figure_to_data_uri(fig),
    }


def _resolve_relationship(
    metadata: Dict[str, Any],
    parent: str,
    child: str,
) -> Optional[Tuple[str, str, str, str]]:
    parent_meta = metadata.get("tables", {}).get(parent, {})
    card = (parent_meta.get("cardinality", {}) or {}).get(child)
    fk_column = card.get("fk_column") if card else None
    pk_column = parent_meta.get("primary_key")
    if fk_column and pk_column:
        return parent, child, fk_column, pk_column
    raise ValueError(f"No cardinality relationship found for {parent} -> {child}")


def _choose_relationship(metadata: Dict[str, Any], rng: np.random.Generator) -> Optional[Tuple[str, str, str, str]]:
    candidates: List[Tuple[str, str, str, str]] = []
    for parent, tmeta in metadata.get("tables", {}).items():
        for child, card in (tmeta.get("cardinality", {}) or {}).items():
            fk_column = card.get("fk_column")
            pk_column = tmeta.get("primary_key")
            if fk_column and pk_column:
                candidates.append((parent, child, fk_column, pk_column))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _relationship_summary_rows(metadata: Dict[str, Any], real: TableMap, synth: TableMap) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parent, tmeta in metadata.get("tables", {}).items():
        pk = tmeta.get("primary_key")
        for child, card in (tmeta.get("cardinality", {}) or {}).items():
            fk_column = card.get("fk_column")
            if not fk_column or parent not in real or child not in real:
                continue
            real_valid = _fk_valid_rate(real[parent][pk], real[child][fk_column]) if pk else None
            synth_valid = _fk_valid_rate(synth[parent][pk], synth[child][fk_column]) if pk else None
            rows.append(
                {
                    "parent_table": parent,
                    "child_table": child,
                    "parent_pk": pk,
                    "child_fk": fk_column,
                    "mean_child_rows": card.get("mean"),
                    "min": card.get("min"),
                    "max": card.get("max"),
                    "p50": card.get("p50"),
                    "p90": card.get("p90"),
                    "real_fk_valid_rate": real_valid,
                    "synthetic_fk_valid_rate": synth_valid,
                }
            )
    return rows


def _fk_valid_rate(parent_keys: pd.Series, child_fk: pd.Series) -> float:
    parent_set = set(parent_keys.dropna().astype(str))
    child_vals = child_fk.dropna().astype(str)
    if len(child_vals) == 0:
        return float("nan")
    return float(child_vals.isin(parent_set).mean())


def _build_html_document(
    title: str,
    results: Dict[str, Any],
    figures: List[Dict[str, Any]],
    real: TableMap,
    synth: TableMap,
    metadata: Dict[str, Any],
) -> str:
    metrics_html = _render_metrics(results)
    overview_html = _render_overview(results, real, synth, metadata)
    relationships_html = _render_relationship_summary(metadata, real, synth)
    figures_html = "".join(_render_figure_card(fig) for fig in figures)
    logo_uri = _logo_data_uri()
    # show white logo by applying a CSS filter; if you have a white PNG/SVG prefer that instead
    logo_img = (
        f'<img src="{logo_uri}" alt="SeRP" style="height:80px; display:block; filter:brightness(0) invert(1);">'
        if logo_uri
        else ""
    )

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
        --muted: #4d5963;
        --accent: {SEPR_BLUE};
        --accent-2: {SEPR_YELLOW};
        --accent-3: #2a7a6b;
        --border: rgba(255, 255, 255, 0.08);
        --shadow: 0 18px 46px rgba(10, 20, 30, 0.06);
    }}
    * {{ box-sizing: border-box; }}
        body {{
        margin: 0;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
        color: var(--ink);
            line-height: 1.5;
        background: var(--bg);
        }}
    .page {{ max-width: 1300px; margin: 0 auto; padding: 28px 20px 48px; }}
        .hero {{
            background: transparent;
        color: var(--panel-strong);
            border-radius: 6px;
            padding: 18px 8px;
        box-shadow: none;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 20px;
        }}
        .hero h1 {{ margin: 0 0 2px; font-size: 1.95rem; letter-spacing: 0.01em; }}
        .hero p {{ margin: 0; max-width: 84ch; color: rgba(255,255,255,0.92); line-height: 1.45; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(12, 1fr); }}
        .card {{
                background: var(--panel-strong);
                border: 1px solid rgba(0,0,0,0.06);
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            }}
    .card .body {{ padding: 18px 18px 20px; }}
    .card h2, .card h3 {{ margin: 0 0 10px; }}
    .card h2 {{
        font-size: 1.35rem;
        color: var(--accent-2);
        background: var(--accent);
        padding: 10px 14px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 10px;
    }}
    .card h3 {{ font-size: 1.1rem; }}
    .muted {{ color: var(--muted); }}
    .summary {{ grid-column: span 12; }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
        .metric-tile {{
        border: 1px solid rgba(0,0,0,0.04);
            border-radius: 10px;
        padding: 12px;
            background: transparent;
        }}
    .metric-tile .label {{ font-size: 0.92rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
    .metric-tile .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .section {{ grid-column: span 12; }}
    .section-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .wide {{ grid-column: span 12; }}
    .figure {{ overflow: hidden; }}
    .figure img {{ width: 100%; display: block; border-top: 1px solid rgba(0,0,0,0.04); background: var(--panel-strong); }}
    .figure .caption {{ padding-top: 8px; font-size: 0.95rem; color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; border-radius: 14px; border: 1px solid var(--border); }}
        table {{ border-collapse: collapse; width: 100%; background: var(--panel-strong); }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee6d7; text-align: left; vertical-align: top; }}
        th {{ background: #f1f5fb; position: sticky; top: 0; z-index: 1; }}
        tr:nth-child(even) td {{ background: rgba(42, 91, 215, 0.03); }}
        details {{ border: 1px solid var(--border); border-radius: 16px; padding: 10px 12px; background: #fff; }}
    details + details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
        .footer {{ margin-top: 24px; color: var(--panel-strong); font-size: 0.92rem; text-align: center; opacity:0.92; }}
    @media (max-width: 980px) {{
      .section-grid {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 1.8rem; }}
    }}
  </style>
</head>
<body>
        <main class="page">
        <header class="hero">
            <div style="display:flex; align-items:center; gap:18px;">
                {logo_img}
                <div>
                    <h1 style="color: var(--accent-2);">{html.escape(title)}</h1>
                    <p>
                        Automated evaluation of the real and synthetic tables, including quality,
                        diagnostic, utility, and privacy scores, plus a visual audit of marginal
                        distributions, scatter structure, and inter-table relationships.
                    </p>
                </div>
            </div>
        </header>
    <section class="grid">
      <div class="card summary">
        <div class="body">
          <h2>Overview</h2>
          {overview_html}
        </div>
      </div>

      <div class="card section wide">
        <div class="body">
          <h2>Evaluation Scores</h2>
          {metrics_html}
        </div>
      </div>

      <div class="card section wide">
        <div class="body">
          <h2>Inter-table Relationships</h2>
          {relationships_html}
        </div>
      </div>

      <div class="card section wide">
        <div class="body">
          <h2>Visual Audit</h2>
          <div class="section-grid">
            {figures_html}
          </div>
        </div>
      </div>
    </section>
        <div class="footer"></div>
  </main>
</body>
</html>"""


def _render_overview(
    results: Dict[str, Any],
    real: TableMap,
    synth: TableMap,
    metadata: Dict[str, Any],
) -> str:
    qualities = results.get("quality", {})
    diagnostic = results.get("diagnostic", {})
    utility = results.get("utility", {})
    privacy = results.get("privacy", {})
    row_count = sum(len(df) for df in real.values())
    synth_rows = sum(len(df) for df in synth.values())
    table_count = len(real)
    relation_count = sum(len((tmeta.get("cardinality", {}) or {})) for tmeta in metadata.get("tables", {}).values())

    tiles = [
        ("Tables", str(table_count)),
        ("Real Rows", str(row_count)),
        ("Synthetic Rows", str(synth_rows)),
        ("Relationships", str(relation_count)),
        ("Quality", _fmt_score(qualities.get("overall_score"))),
        ("Diagnostic", _fmt_score(diagnostic.get("overall_score"))),
        ("Utility", _fmt_score(utility.get("overall_score"))),
        ("Privacy", _fmt_score(privacy.get("overall_score"))),
    ]
    tiles_html = "".join(
        f'<div class="metric-tile"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in tiles
    )
    return f'<div class="summary-grid">{tiles_html}</div>'


_SECTION_INFO: Dict[str, Dict[str, Any]] = {
    "quality": {
        "summary": (
            "Measures how closely the synthetic data reproduces the statistical properties "
            "of the real data — marginal distributions, pairwise trends, and cross-table relationships."
        ),
        "interpretation": "Score ranges 0–1. Above 0.85 is generally considered good. Higher is better.",
        "metrics": [
            ("Column Shapes", "Compares the marginal distribution of each column. KS complement for numeric columns, Total Variation complement for categorical. Score 1.0 = identical distributions."),
            ("Column Pair Trends", "Compares pairwise correlations and associations. Captures how well joint distributions and feature interactions are preserved."),
            ("Cardinality", "Checks that the number of child rows per parent (e.g. conditions per patient) follows a similar distribution in real and synthetic data."),
            ("Intertable Trends", "Compares cross-table correlations (e.g. patient birthdate vs condition start date). Captures whether inter-table dependencies are preserved."),
        ],
    },
    "diagnostic": {
        "summary": (
            "Validates the structural integrity of the synthetic data — primary key uniqueness, "
            "foreign key referential validity, and that values stay within observed ranges."
        ),
        "interpretation": "Score ranges 0–1 and should be 1.0 for a correctly generated dataset. Any score below 1.0 indicates a structural problem worth investigating.",
        "metrics": [],
    },
    "utility": {
        "summary": (
            "Measures how indistinguishable the synthetic data is from the real data to a logistic "
            "regression classifier trained to tell them apart."
        ),
        "interpretation": (
            "Score = 1 − 2 | AUC − 0.5 |. "
            "1.0 = perfectly indistinguishable (maximum utility). "
            "0.0 = perfectly separable (no utility). "
            "Scores ≥0.8 indicate the synthetic data is a strong substitute for the real data."
        ),
        "metrics": [
            ("logistic_detection", "Per-table score. Higher means the synthetic data is more useful as a stand-in for the real data."),
        ],
    },
    "privacy": {
        "summary": (
            "Assesses the risk that synthetic records could be linked back to real individuals. "
            "The overall score averages three complementary signals; higher is better."
        ),
        "interpretation": (
            "All sub-metrics are higher-is-better except <em>privacy_risk_fraction</em> (lower is safer). "
            "An overall score above 0.7 is generally acceptable for de-identified research data."
        ),
        "metrics": [
            ("score", "Per-table aggregate privacy score (0–1). Mean of new_row_synthesis, nndr_above_0.5, and (1 − privacy_risk_fraction)."),
            ("new_row_synthesis", "Fraction of synthetic rows that do not exactly duplicate any real row. 1.0 = no verbatim copies."),
            ("nn_distance_median", "Median distance from each synthetic row to its nearest real neighbour, computed in mixed (numeric + categorical) feature space. Higher = synthetic data sits further from real records."),
            ("nndr_median", "Median Nearest Neighbour Distance Ratio: d₁/d₂ where d₁ is the distance to the closest real row and d₂ to the second closest. Values near 1.0 mean no single real record is disproportionately close to a synthetic one."),
            ("nndr_above_0.5", "Proportion of synthetic rows with NNDR ≥0.5. Higher = fewer synthetic records with a uniquely-close real counterpart (lower singling-out risk)."),
            ("privacy_risk_fraction", "Fraction of synthetic rows closer to any real row than the 5th-percentile of real-to-real distances (a data-driven threshold). Lower = safer."),
        ],
    },
}


def _render_section_description(section_name: str) -> str:
    info = _SECTION_INFO.get(section_name, {})
    summary = info.get("summary", "")
    interpretation = info.get("interpretation", "")
    metrics = info.get("metrics", [])

    if not summary and not interpretation:
        return ""

    metric_rows = "".join(
        f'<tr><td style="white-space:nowrap;font-weight:600;padding:5px 10px 5px 0;">{html.escape(name)}</td>'
        f'<td style="padding:5px 0;">{html.escape(desc)}</td></tr>'
        for name, desc in metrics
    )
    metric_table = (
        f'<table style="border-collapse:collapse;width:100%;margin-top:6px;">{metric_rows}</table>'
        if metric_rows else ""
    )

    return (
        f'<p style="color:#4d5963;margin:0 0 8px;font-size:0.97rem;">{html.escape(summary)}</p>'
        f'<details style="margin-bottom:12px;border:1px solid #dde4ef;border-radius:8px;padding:8px 12px;background:#f7f9fc;">'
        f'<summary style="cursor:pointer;font-weight:600;color:#003761;font-size:0.93rem;">How to interpret</summary>'
        f'<div style="padding:8px 0 2px;font-size:0.93rem;color:#2c3e50;">'
        f'<p style="margin:0 0 6px;">{interpretation}</p>'
        f'{metric_table}'
        f'</div>'
        f'</details>'
    )


def _render_metrics(results: Dict[str, Any]) -> str:
    sections = []
    for section_name in ("quality", "diagnostic", "utility", "privacy"):
        section = results.get(section_name, {})
        sections.append(_render_metric_section(section_name, section))
    return "".join(sections)


def _render_metric_section(section_name: str, section: Dict[str, Any]) -> str:
    title = section_name.title()
    score = section.get("overall_score")
    score_html = f"<strong>{_fmt_score(score)}</strong>" if score is not None else "n/a"
    description_html = _render_section_description(section_name)
    details_html = []

    properties = section.get("properties", []) or []
    if properties:
        details_html.append(_render_dataframe(properties, max_rows=50))

    for prop_name, rows in (section.get("details", {}) or {}).items():
        details_html.append(
            f"<details><summary>{html.escape(prop_name)}</summary>{_render_dataframe(rows, max_rows=75)}</details>"
        )

    if section_name == "utility":
        details_html.insert(0, _render_table_rows(_utility_rows(section)))
    elif section_name == "privacy":
        details_html.insert(0, _render_table_rows(_privacy_rows(section)))

    body = "".join(details_html) if details_html else '<p class="muted">No breakdown available.</p>'
    return (
        f'<section class="card" style="margin-top:12px;">'
        f'<div class="body">'
        f'<h3>{html.escape(title)}: {score_html}</h3>'
        f'{description_html}'
        f'{body}'
        f'</div></section>'
    )


def _utility_rows(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table, payload in section.items():
        if table == "overall_score":
            continue
        if isinstance(payload, dict):
            rows.append({
                "table": table,
                "logistic_detection": payload.get("logistic_detection"),
                "error": payload.get("error"),
            })
    return rows


def _privacy_rows(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table, payload in section.items():
        if table == "overall_score":
            continue
        if isinstance(payload, dict):
            nn = payload.get("nn_distance") or {}
            nndr = payload.get("nndr") or {}
            rows.append({
                "table": table,
                "score": payload.get("score"),
                "new_row_synthesis": payload.get("new_row_synthesis"),
                "nn_distance_median": nn.get("median"),
                "nndr_median": nndr.get("median"),
                "nndr_above_0.5": nndr.get("proportion_above_0_5"),
                "privacy_risk_fraction": payload.get("privacy_risk_fraction"),
                "error": payload.get("new_row_synthesis_error"),
            })
    return rows


def _render_relationship_summary(metadata: Dict[str, Any], real: TableMap, synth: TableMap) -> str:
    rows = _relationship_summary_rows(metadata, real, synth)
    if not rows:
        return '<p class="muted">No foreign key relationships were found in the metadata.</p>'
    return _render_dataframe(rows, max_rows=100)


def _render_figure_card(fig: Dict[str, Any]) -> str:
    return (
        '<div class="card figure">'
        '<div class="body">'
        f'<h3>{html.escape(fig["title"])}</h3>'
        f'<div class="caption">{html.escape(fig.get("caption", ""))}</div>'
        f'<img src="{fig["image"]}" alt="{html.escape(fig["title"])}" />'
        '</div>'
        '</div>'
    )


def _render_dataframe(rows: Any, max_rows: int = 50) -> str:
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows)
    if df.empty:
        return '<p class="muted">No rows.</p>'
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
    return f'<div class="table-wrap">{df.to_html(index=False, escape=True, border=0)}</div>'


def _render_table_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    return _render_dataframe(rows, max_rows=100)


def _fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML report for AusSynth evaluations.")
    parser.add_argument("--real", required=True, help="Real tables: directory of CSVs or JSON mapping file")
    parser.add_argument("--synthetic", required=True, help="Synthetic tables: directory of CSVs or JSON mapping file")
    parser.add_argument("--metadata", required=True, help="Metadata JSON produced by AusSynthPackage.process")
    parser.add_argument("--output", default="report.html", help="Output HTML path")
    parser.add_argument("--title", default="AusSynth Synthetic Data Report", help="Report title")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for visual selection")
    parser.add_argument("--exclude-columns", default=None, help="Optional JSON file mapping table -> columns to exclude")
    args = parser.parse_args(argv)

    exclude_columns = None
    if args.exclude_columns:
        with open(args.exclude_columns, "r", encoding="utf-8") as f:
            exclude_columns = json.load(f)

    generate_html_report(
        real_tables=args.real,
        synthetic_tables=args.synthetic,
        metadata=args.metadata,
        output_path=args.output,
        title=args.title,
        seed=args.seed,
        exclude_columns=exclude_columns,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())