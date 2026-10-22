"""Compare a fresh run with the numerical tables used for the manuscript.

This checks reproducibility against a fixed published target, rather than testing
that the program repeats its own formulas. It defines no classes. References
contain published synthetic outputs, not field observations. Only NumPy and
pandas are needed; rendering verification is performed for the release separately.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_results(output_directory, reference_directory):
    """Compare all CSV shapes, labels and values; write a machine-readable report.

    Floating-point tolerance permits small numerical-library/platform effects.
    Discrete identifiers, strings, sample counts and row order must still match.
    A mismatch raises an error, so a changed experiment is never reported as a
    successful reproduction of the manuscript's tables.
    """
    output_directory = Path(output_directory)
    reference_directory = Path(reference_directory)
    comparisons = []
    for reference_path in sorted(reference_directory.glob("*.csv")):
        fresh_path = output_directory / reference_path.name
        if not fresh_path.is_file():
            raise FileNotFoundError(f"Missing regenerated table: {fresh_path}")
        reference = pd.read_csv(reference_path)
        fresh = pd.read_csv(fresh_path)
        if (
            list(fresh.columns) != list(reference.columns)
            or fresh.shape != reference.shape
        ):
            raise ValueError(f"Changed table structure: {reference_path.name}")
        maximum_error = 0.0
        for column in reference.columns:
            if pd.api.types.is_numeric_dtype(reference[column]):
                expected = reference[column].to_numpy(dtype=float)
                actual = fresh[column].to_numpy(dtype=float)
                if not np.allclose(
                    actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
                ):
                    raise ValueError(
                        f"Numerical mismatch: {reference_path.name}/{column}"
                    )
                valid = np.isfinite(actual) & np.isfinite(expected)
                if valid.any():
                    maximum_error = max(
                        maximum_error,
                        float(np.max(abs(actual[valid] - expected[valid]))),
                    )
            elif not fresh[column].fillna("").equals(reference[column].fillna("")):
                raise ValueError(f"Changed labels: {reference_path.name}/{column}")
        comparisons.append(
            {
                "file": reference_path.name,
                "rows": len(fresh),
                "maximum_absolute_error": maximum_error,
                "passed": True,
            }
        )
    report = {
        "passed": True,
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-12,
        "tables": comparisons,
    }
    (output_directory / "verification_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        f"Verified {len(comparisons)} CSV tables against manuscript reference results."
    )
    return report
