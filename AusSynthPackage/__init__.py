"""
synth_data: Generate synthetic data preserving marginal distributions
            while removing inter-variable relationships.

Two main entry points:
    - process(tables, output_path=None, **kwargs) -> metadata dict
        Inspect a relational dataset and produce SDC-safe metadata.
    - generate(metadata, n_rows=None, output_dir=None) -> dict of DataFrames
        Sample synthetic tables from the metadata.
"""

from .process import metadata_to_dataframe, metadata_to_html, process
from .generate import generate
from .evaluate import evaluate, find_suppressed_columns
from .report import generate_html_report

__all__ = [
    "process",
    "metadata_to_dataframe",
    "metadata_to_html",
    "generate",
    "evaluate",
    "find_suppressed_columns",
    "report",
    "generate_html_report"
]
__version__ = "0.1.0"
