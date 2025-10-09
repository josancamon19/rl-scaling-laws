#!/usr/bin/env python3
"""
Clean export.csv by keeping only mean@1 columns and simplifying column names.
"""

import csv
import re
from pathlib import Path


def clean_export_csv(input_path: str, output_path: str = None):
    """
    Clean the export CSV by:
    1. Keeping only mean@1 columns (not MIN or MAX)
    2. Simplifying column names to show only model size (14b, 8b, 4b, 1.7b)

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save cleaned CSV (defaults to input_path with '_cleaned' suffix)
    """
    # Read the CSV
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    # Keep Step column and columns that end with mean@1 (not __MIN or __MAX)
    columns_to_keep = []
    new_column_names = []

    for col in headers:
        if col == "Step":
            columns_to_keep.append(col)
            new_column_names.append(col)
            continue

        # Check if it's a mean@1 column (not MIN or MAX)
        if "mean@1" in col and "__MIN" not in col and "__MAX" not in col:
            columns_to_keep.append(col)

            # Extract model size (14b, 8b, 4b, 1.7b, etc.)
            # Look for pattern after "qwen" version, like qwen3_14b or qwen3_1_7b
            match = re.search(r"qwen\d+_(\d+(?:_\d+)?b)", col)
            if match:
                model_size = match.group(1).replace("_", ".")  # Normalize 1_7b to 1.7b
                new_column_names.append(model_size)
            else:
                new_column_names.append(col)

    # Determine output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = (
            input_file.parent / f"{input_file.stem}{input_file.suffix}"
        )

    # Write cleaned CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(new_column_names)

        # Write data rows
        for row in rows:
            cleaned_row = [row[col] for col in columns_to_keep]
            writer.writerow(cleaned_row)

    print(f"Cleaned CSV saved to: {output_path}")
    print(f"Columns kept: {new_column_names}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    import sys

    # Default path
    default_input = Path(__file__).parent.parent / "results" / "export.csv"

    # Allow command line argument for custom input path
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    clean_export_csv(input_path, output_path)
