"""Appendices B and C: conversion and historical discrete-flow checks."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import (
    lhs,
    outputs,
    diagnostics,
    ALPHA,
    COEFF,
    BOUNDS,
    NAMES,
    uniform_log_variance,
)
from plots import slope_schematic

# US quantities equal these factors times their corresponding SI quantities.
FLOW_US = np.array([40.0, 110.0, 190.0, 260.0])
US_PER_SI = np.array(
    [1 / 0.3048, 1 / 0.40468564224, 1.0, 1 / 0.0254, 1.0, 1.0, 1 / 0.003785411784]
)
COEFF_US = np.array([0.0038, 0.001303, 0.00389, 0.0076])


def discrete_flow(n=200000, seed=123):
    """Replace only baseline flow with the four historical levels, design F.

    The other six columns match the baseline exactly. Equal quarter-intervals
    of the seventh LHS coordinate choose 40, 110, 190 or 260 US gal/person/day.
    Exact US-gallon conversion is applied before calculating outputs. For the
    paper's N=200000, every level receives exactly 50000 rows."""
    x = lhs(n, seed)
    u = (x[:, 6] - BOUNDS[6, 0]) / (BOUNDS[6, 1] - BOUNDS[6, 0])
    x[:, 6] = FLOW_US[np.minimum((4 * u).astype(int), 3)] * 0.003785411784
    return x


def run(out, n=200000):
    """Compute design F, exact discrete-flow shares and the US--SI identity checks.

    The twelve conversion checks use the same rounded exponent vector on both
    sides. They compare reconstructed SI coefficients against historical US
    coefficients, while separately reporting the retained audited SI rounding.
    The common d**(-0.510) factor cancels in every same-row pairwise ratio.
    Finally, draw Appendix Figure A1 and save the numerical audit manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = lhs(n, 123)
    f = discrete_flow(n, 123)
    yb = outputs(b)
    yf, s, p, sm = diagnostics(f, "F")
    pd.DataFrame([sm]).to_csv(out / "flow_summary.csv", index=False)
    s.to_csv(out / "flow_sensitivity.csv", index=False)
    p.to_csv(out / "flow_pairwise.csv", index=False)
    vv = uniform_log_variance(*BOUNDS.T)
    # Only this marginal variance changes in the exact mixed-input formula.
    vv[6] = np.var(np.log10(FLOW_US * 0.003785411784), ddof=0)
    shares = ALPHA**2 * vv
    shares /= shares.sum(axis=1)[:, None]
    rows = []
    for k in range(4):
        for v in np.flatnonzero(ALPHA[k]):
            rows.append(
                dict(design="F", equation=k + 1, input=NAMES[v], share=shares[k, v])
            )
    pd.DataFrame(rows).to_csv(out / "flow_exact_shares.csv", index=False)
    counts = np.unique(f[:, 6], return_counts=True)
    pd.DataFrame(dict(flow_US=FLOW_US, flow_SI=counts[0], count=counts[1])).to_csv(
        out / "flow_levels.csv", index=False
    )
    # Transform coefficients along with all powers of the input unit factors.
    converted = 0.45359237 * COEFF_US * np.exp(ALPHA @ np.log(US_PER_SI))
    points = [BOUNDS[:, 0], np.sqrt(BOUNDS.prod(axis=1)), BOUNDS[:, 1]]
    cross = []
    max_rel = 0.0
    for label, x in zip(["lower_corner", "geometric_midpoint", "upper_corner"], points):
        us = COEFF_US * np.exp(ALPHA @ np.log(x * US_PER_SI))
        si = converted * np.exp(ALPHA @ np.log(x))
        aud = 10 ** outputs(x)
        for k in range(4):
            rel = abs(si[k] / (0.45359237 * us[k]) - 1)
            max_rel = max(max_rel, rel)
            cross.append(
                dict(
                    point=label,
                    equation=k + 1,
                    US_output_lb_day=us[k],
                    US_output_converted_kg_day=0.45359237 * us[k],
                    SI_output_kg_day=si[k],
                    relative_identity_error=rel,
                    audited_SI_kg_day=aud[k],
                    audited_vs_converted_percent=100 * (aud[k] / si[k] - 1),
                )
            )
    pd.DataFrame(cross).to_csv(out / "US_SI_crosscheck.csv", index=False)
    pd.DataFrame(
        points,
        columns=NAMES,
        index=["lower_corner", "geometric_midpoint", "upper_corner"],
    ).to_csv(out / "US_SI_test_inputs.csv", index_label="point")
    # Flow is shared across equations, so it cancels in every paired log ratio.
    invariant = max(
        (
            np.max(abs(yf[:, i] - yf[:, j] - (yb[:, i] - yb[:, j])))
            for i in range(4)
            for j in range(i + 1, 4)
        )
    )
    assert np.all(counts[1] == n // 4) and n % 4 == 0
    assert max_rel < 1e-12 and invariant < 1e-12
    slope_schematic(out)
    meta = {
        "date": "2026-10-21",
        "n": n,
        "seed": 123,
        "flow_level_weights": [0.25] * 4,
        "flow_SI": (FLOW_US * 0.003785411784).tolist(),
        "max_US_SI_relative_error": max_rel,
        "max_pairwise_logratio_change_due_to_flow": invariant,
        "scope": "F changes only common wastewater flow; other six columns match B exactly.",
    }
    (out / "reviewer3_manifest.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame([sm]).to_string(index=False))
    print(p.to_string(index=False))
    print(json.dumps(meta, indent=2))
