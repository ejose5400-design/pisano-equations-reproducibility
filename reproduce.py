"""One-command reproduction of all manuscript results and figures.

Run from this directory: python reproduce.py
For only the figures:   python reproduce.py --figures-only
For sampled matrices:  python reproduce.py --save-samples
"""

import sys
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib
from model import (
    NAMES,
    BOUNDS,
    ALPHA,
    COEFF,
    FIELD_L,
    FIELD_A,
    FIELD_S,
    DENSITY,
    lhs,
    outputs,
    diagnostics,
    constrained,
    narrowed,
    uniform_log_variance,
    reordering_tables,
)
from plots import figures, slope_schematic
from extra_checks import run as run_extra_checks
from verify_results import verify_results


def run_base_analysis(n, out, save_samples=False):
    """Run B, O, C, E, H, HL, HP and G with their declared seeds.

    This function exports all baseline diagnostics, field context, analytical
    shares and convergence replications. Optional NPZ files contain each sampled
    input matrix and its log outputs; omit them from GitHub because they can be
    regenerated from this script."""
    out.mkdir(parents=True, exist_ok=True)
    # All designs share the same variable order and fixed published sample size.
    designs = {"B": lhs(n, 123)}
    meta = {}
    designs["O"], meta["O"] = constrained(n, 223)
    designs["C"], meta["C"] = constrained(n, 323, density=True)
    designs["E"], meta["E"] = constrained(n, 423, exponential=True, density=True)
    designs["H"] = lhs(n, 123, bounds=narrowed(np.arange(7)))
    designs["HL"] = lhs(n, 123, bounds=narrowed(np.array([0])))
    designs["HP"] = lhs(n, 123, bounds=narrowed(np.array([4])))
    designs["G"] = lhs(n, 123, log_uniform=True)
    # Store design-level summaries and equation/input or equation/pair tables.
    summary = []
    senss = []
    pairss = []
    for name, x in designs.items():
        print("Design", name, flush=True)
        y, s, p, sm = diagnostics(x, name)
        summary.append(sm)
        senss.append(s)
        pairss.append(p)
        if save_samples:
            np.savez_compressed(out / f"sample_{name}.npz", inputs=x, log_outputs=y)
        if name == "B":
            figures(x, y, s, p, out)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(senss).to_csv(out / "sensitivity.csv", index=False)
    pd.concat(pairss).to_csv(out / "pairwise.csv", index=False)
    # Compute population independent log shares analytically, not from samples.
    analytic = []
    for name, b in [
        ("B", BOUNDS),
        ("H", narrowed(np.arange(7))),
        ("HL", narrowed(np.array([0]))),
        ("HP", narrowed(np.array([4]))),
    ]:
        vv = uniform_log_variance(*b.T)
        raw = ALPHA**2 * vv
        share = raw / raw.sum(axis=1)[:, None]
        for k in range(4):
            for v in np.flatnonzero(ALPHA[k]):
                analytic.append(
                    dict(
                        design=name,
                        equation=k + 1,
                        input=NAMES[v],
                        elasticity=ALPHA[k, v],
                        share=share[k, v],
                        range_decades=ALPHA[k, v] * np.log10(b[v, 1] / b[v, 0]),
                    )
                )
    pd.DataFrame(analytic).to_csv(out / "analytic.csv", index=False)
    reordering_tables(designs, out)
    # External-domain field context: these published inputs do not validate EQ4.
    field_eq4 = (
        COEFF[3]
        * FIELD_L ** ALPHA[3, 0]
        * FIELD_S ** ALPHA[3, 2]
        * 0.149 ** ALPHA[3, 6]
    )
    fit = 0.0742 * FIELD_S ** (-0.509)
    pd.DataFrame(
        dict(
            catchment=[
                "Infante",
                "Faroliva",
                "El Ferrocarril",
                "Central",
                "Algezares II",
                "Plaza Circular",
                "Patino",
                "Juan XXIII",
            ],
            L=FIELD_L,
            A=FIELD_A,
            S=FIELD_S,
            EQ4=field_eq4,
            EQ4_per_m=field_eq4 / FIELD_L,
            published_fitted_field_per_m=fit,
            fitted_field_total_proxy=fit * FIELD_L,
        )
    ).to_csv(out / "field_context.csv", index=False)
    # Three independent seeds at each size quantify numerical replication.
    convergence = []
    for nn in [10000, 50000, n]:
        for seed in [123, 124, 125]:
            _, _, pp, ss = diagnostics(lhs(nn, seed), "check")
            ss.update(
                seed=seed,
                rho14=pp.query("i==1 and j==4").spearman.iloc[0],
                overlap14=pp.query("i==1 and j==4").top_decile_overlap.iloc[0],
            )
            convergence.append(ss)
    pd.DataFrame(convergence).to_csv(out / "convergence.csv", index=False)
    meta.update(
        n=n,
        baseline_seed=123,
        variables=NAMES,
        bounds=BOUNDS.tolist(),
        coefficients=COEFF.tolist(),
        exponents=ALPHA.tolist(),
        density_bounds=DENSITY,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        matplotlib=matplotlib.__version__,
    )
    pd.DataFrame(
        dict(
            equation=np.arange(1, 5),
            historical_US=[0.0038, 0.001303, 0.00389, 0.0076],
            converted_SI=0.45359237
            * np.array([0.0038, 0.001303, 0.00389, 0.0076])
            * np.exp(
                ALPHA
                @ np.log(
                    [
                        1 / 0.3048,
                        1 / 0.40468564224,
                        1,
                        1 / 0.0254,
                        1,
                        1,
                        1 / 0.003785411784,
                    ]
                )
            ),
            audited_SI=COEFF,
        )
    ).to_csv(out / "coefficient_audit.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    assert all((np.all(v > 0) for v in designs.values()))
    assert all((np.all(designs[k][:, 5] <= designs[k][:, 4]) for k in ["O", "C", "E"]))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    """Use the fixed manuscript design; only output location/export mode changes."""
    arguments = sys.argv[1:]
    permitted = {"--figures-only", "--save-samples", "--out"}
    unknown = [a for a in arguments if a.startswith("--") and a not in permitted]
    if unknown:
        raise SystemExit("Unknown option(s): " + ", ".join(unknown))
    root = Path(__file__).resolve().parent
    output_directory = root / "results"
    if "--out" in arguments:
        position = arguments.index("--out")
        if position + 1 == len(arguments):
            raise SystemExit("--out requires a directory path")
        output_directory = Path(arguments[position + 1]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sample_size = 200000  # Fixed published experiment; do not silently downsample.
    if "--figures-only" in arguments:
        inputs = lhs(sample_size, 123)
        log_mass, sensitivity, pairs, summary = diagnostics(inputs, "B")
        figures(inputs, log_mass, sensitivity, pairs, output_directory)
        slope_schematic(output_directory)
        print("Regenerated Figures 1--5 and Appendix Figure A1.")
    else:
        run_base_analysis(sample_size, output_directory, "--save-samples" in arguments)
        run_extra_checks(output_directory, sample_size)
        verify_results(output_directory, root / "reference_results")
    print("Output directory:", output_directory)
    print("Elapsed seconds:", round(time.perf_counter() - start, 1))


if __name__ == "__main__":
    main()
