#!/usr/bin/env bash
# =============================================================================
# describe_csvs.sh
#
# Recursively finds every .csv file under the uav4slb repository root and
# writes a human-readable data dictionary (headers, dtypes, shape, value
# ranges, missing-value counts, and a 3-row preview) to a single .txt report.
#
# The report is grouped by top-level subdirectory (data/, experiments/,
# figures/, results/) for easier navigation.
#
# Usage:
#   bash describe_csvs.sh [REPO_ROOT] [OUTPUT_FILE]
#
# Defaults (if no arguments are given):
#   REPO_ROOT   = directory containing this script (i.e. the repo root)
#   OUTPUT_FILE = <REPO_ROOT>/data_csv_descriptions.txt
#
# Requirements:
#   Python 3 with pandas available on PATH.
#   Activate the project conda environment before running:
#     conda activate uav_for_slb
#
# Example:
#   conda activate uav_for_slb
#   bash scripts/describe_csvs.sh \
#       /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/uav4slb \
#       /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/uav4slb/data_csv_descriptions.txt
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Argument handling
# ---------------------------------------------------------------------------

# Repo root: use first CLI arg, or the directory that contains this script.
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Output file: use second CLI arg, or place report at the repo root.
OUTPUT_FILE="${2:-${REPO_ROOT}/data_csv_descriptions.txt}"

echo "Repository root : ${REPO_ROOT}"
echo "Output file     : ${OUTPUT_FILE}"

# ---------------------------------------------------------------------------
# 2. Verify pandas is importable (friendly error message if env not active)
# ---------------------------------------------------------------------------
if ! python3 -c "import pandas" 2>/dev/null; then
    echo "ERROR: pandas not found on the current Python path." >&2
    echo "Activate the project conda environment first:" >&2
    echo "  conda activate uav_for_slb" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Collect all .csv paths under the repo root (sorted, NUL-delimited)
# ---------------------------------------------------------------------------
# Exclude __pycache__ and hidden directories (e.g. .git).
# Store as a newline-delimited list in a temp file to avoid stdin conflicts
# with the Python heredoc.
TMPFILE=$(mktemp)
trap 'rm -f "${TMPFILE}"' EXIT

find "${REPO_ROOT}" \
    -not -path '*/__pycache__/*' \
    -not -path '*/.git/*' \
    -name "*.csv" \
    -print0 \
| sort -z \
| tr '\0' '\n' \
> "${TMPFILE}"

N_FILES=$(wc -l < "${TMPFILE}" | tr -d ' ')
echo "Found ${N_FILES} CSV file(s). Generating descriptions …"

# ---------------------------------------------------------------------------
# 4. Run the Python description script, passing the temp file path as an arg
# ---------------------------------------------------------------------------
python3 - "${REPO_ROOT}" "${OUTPUT_FILE}" "${TMPFILE}" <<'PYEOF'
"""
Inline Python script: reads CSV paths (one per line) from a temp file,
inspects each file with pandas, and writes a structured plain-text data
dictionary to OUTPUT_FILE.

Report structure:
  • Document header   – timestamp, repo root, file count
  • Table of contents – section names and file counts
  • Per-section block – one block per top-level subdirectory
  • Per-file block    – relative path, size, shape, column table, 3-row preview

Column table fields:
  COLUMN_NAME | DTYPE | NULLS (count + %) | RANGE or TOP_VALUES
"""

import sys
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd

# ── CLI arguments injected by the outer bash script ─────────────────────────
REPO_ROOT   = sys.argv[1]   # absolute path to the repository root
OUTPUT_FILE = sys.argv[2]   # absolute path for the output .txt file
TMPFILE     = sys.argv[3]   # path to newline-delimited list of CSV paths

# ── Read CSV paths from the temp file ────────────────────────────────────────
with open(TMPFILE, "r") as f:
    csv_paths = [line.rstrip("\n") for line in f if line.strip()]


# ── Helper functions ──────────────────────────────────────────────────────────

def relative(path: str) -> str:
    """Return path relative to REPO_ROOT for display; fall back to absolute."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def dtype_label(series: pd.Series) -> str:
    """
    Map a pandas dtype to a concise human-readable label:
      int / float / bool / datetime / str / mixed
    Inspects up to 200 non-null values for object-dtype columns to
    distinguish uniform string columns from mixed-type columns.
    """
    kind = series.dtype.kind
    if kind in ("i", "u"):
        return "int"
    if kind == "f":
        return "float"
    if kind == "b":
        return "bool"
    if kind == "M":
        return "datetime"
    if kind == "O":
        non_null = series.dropna()
        if len(non_null) == 0:
            return "str (all null)"
        if all(isinstance(v, str) for v in non_null.iloc[:200]):
            return "str"
        return "mixed"
    return str(series.dtype)


def numeric_stats(series: pd.Series) -> str:
    """
    Return a compact one-line range summary for numeric columns:
      min=X  max=Y  mean=Z
    Uses g-format so both very small and very large values display cleanly.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return "no non-null values"
    return (
        f"min={clean.min():.4g}  "
        f"max={clean.max():.4g}  "
        f"mean={clean.mean():.4g}"
    )


def categorical_stats(series: pd.Series, max_cats: int = 8) -> str:
    """
    Return the most frequent values and their counts for string/object columns.
    Shows up to max_cats entries; appends an overflow note when truncated.
    NaN is included in the value_counts so missing-value categories are visible.
    """
    vc = series.value_counts(dropna=False)
    top = vc.head(max_cats)
    items = [f"'{k}' ({v})" for k, v in top.items()]
    overflow = f"  … (+{len(vc) - max_cats} more)" if len(vc) > max_cats else ""
    return "top values: " + ", ".join(items) + overflow


def describe_csv(path: str) -> str:
    """
    Produce a multi-line description for a single CSV file.

    Layout:
      FILE  : <relative path>
      SIZE  : <KB>
      SHAPE : <rows> rows × <cols> columns

      COLUMNS
      -----------------------------------------------------------------------
        <name>  <dtype>  nulls=N (pct%)  <range or top values>
        ...

      PREVIEW (first 3 rows)
      -----------------------------------------------------------------------
        <header + 3 data rows, capped at 120 chars per line>
    """
    rel     = relative(path)
    size_kb = os.path.getsize(path) / 1024

    # ── Load CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return (
            f"FILE  : {rel}\n"
            f"  ERROR: could not read file — {exc}\n"
        )

    n_rows, n_cols = df.shape

    lines = [
        f"FILE  : {rel}",
        f"SIZE  : {size_kb:.1f} KB",
        f"SHAPE : {n_rows:,} rows × {n_cols} columns",
        "",
        "COLUMNS",
        "-" * 72,
    ]

    # Column-name display width: cap at 35 chars to keep lines manageable
    max_name_width = min(max((len(c) for c in df.columns), default=10), 35)

    for col in df.columns:
        s       = df[col]
        dtype   = dtype_label(s)
        n_null  = int(s.isna().sum())
        pct     = (n_null / n_rows * 100) if n_rows > 0 else 0.0

        # Range summary for numerics; value-count summary for categoricals
        stats = numeric_stats(s) if dtype in ("int", "float") else categorical_stats(s)

        null_str    = f"nulls={n_null} ({pct:.1f}%)"
        name_field  = col[:max_name_width].ljust(max_name_width)
        dtype_field = dtype.ljust(10)

        lines.append(
            f"  {name_field}  {dtype_field}  {null_str:<22}  {stats}"
        )

    # ── 3-row data preview ───────────────────────────────────────────────────
    lines += ["", "PREVIEW (first 3 rows)", "-" * 72]
    try:
        preview = df.head(3).to_string(index=False, max_cols=20)
        for row in preview.splitlines():
            lines.append("  " + row[:120])   # truncate very wide rows
    except Exception:
        lines.append("  (preview unavailable)")

    return "\n".join(lines)


# ── Group files by top-level subdirectory for a navigable report ─────────────
sections: dict = defaultdict(list)
for p in csv_paths:
    rel = os.path.relpath(p, REPO_ROOT)
    # First component of the relative path becomes the section heading.
    # Files directly at the repo root get section key ".".
    top = rel.split(os.sep)[0] if os.sep in rel else "."
    sections[top].append(p)


# ── Write the report ──────────────────────────────────────────────────────────
SEP  = "=" * 80
DASH = "-" * 80

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    # Document header
    out.write(SEP + "\n")
    out.write("CSV DATA DICTIONARY — uav4slb\n")
    out.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.write(f"Repo root : {REPO_ROOT}\n")
    out.write(f"Files     : {len(csv_paths)} CSV files catalogued\n")
    out.write(SEP + "\n\n")

    # Table of contents
    out.write("TABLE OF CONTENTS\n")
    out.write(DASH + "\n")
    for section in sorted(sections):
        n = len(sections[section])
        out.write(f"  {section}/  ({n} file{'s' if n != 1 else ''})\n")
    out.write("\n\n")

    # One section block per top-level subdirectory
    for section in sorted(sections):
        paths = sections[section]

        out.write(SEP + "\n")
        out.write(f"SECTION: {section}/\n")
        out.write(SEP + "\n\n")

        for path in paths:
            out.write(describe_csv(path))
            out.write("\n\n" + DASH + "\n\n")

print(f"Done — report written to:\n  {OUTPUT_FILE}")
PYEOF

echo "describe_csvs.sh completed successfully."
