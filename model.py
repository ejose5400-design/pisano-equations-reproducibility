"""Equations, inputs and analytical/sampled diagnostics for the Pisano audit.

Public-code release 1.0.0, prepared 21 October 2026.
All scientific functions are procedural; no custom classes are defined.
See README.md for units, design assumptions and limitations.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata, spearmanr, kendalltau

# One input order is used for bounds, exponent columns and every sampled row.
# Length is metres, area hectares, diameter metres, slopes dimensionless,
# and flow is m^3 per person per day. Outputs are kg per day.
NAMES = ["L", "A", "S", "D", "SPD", "SPD4", "d"]
LABELS = ["$L$", "$A$", "$S$", "$D$", "$S_{PD}$", "$S_{PD/4}$", "$d$"]
BOUNDS = np.array(
    [
        [80.16, 41004.13],
        [1.214, 259.404],
        [0.00238, 0.0799],
        [0.2032, 0.508],
        [0.0011, 0.0589],
        [8.9e-05, 0.0249],
        [0.1514, 0.9842],
    ]
)
ALPHA = np.array(
    [
        [0.814, 0, 0, 0, -0.819, -0.108, -0.51],
        [1.18, -0.178, -0.418, 0.604, 0, 0, -0.51],
        [1.22, -0.187, -0.434, 0, 0, 0, -0.51],
        [1.063, 0, -0.438, 0, 0, 0, -0.51],
    ]
)
COEFF = np.array([0.000264, 0.001091, 0.000369, 0.000709])
FIELD_L = np.array([44921, 2639, 10354, 91725, 4008, 14599, 9041, 11906.0])
FIELD_A = np.array([175.67, 43.7, 45.42, 450.53, 63.21, 52.5, 87.58, 87.91])
FIELD_S = np.array([0.0006, 0.0013, 0.0014, 0.0018, 0.0037, 0.0043, 0.0062, 0.0113])
DENSITY = (min(FIELD_L / FIELD_A), max(FIELD_L / FIELD_A))

# Positive empirical coefficients and their rounded exponent convention above
# define the equations being audited, not a new observational calibration.


def lhs(n, seed, bounds=BOUNDS, log_uniform=False):
    """Generate one common n-by-7 Latin hypercube with an explicit seed.

    Columns are L, A, S, D, SPD, SPD4 and d, always in that order. Linear scaling
    gives uniform marginals; log scaling gives log-uniform marginals. Selecting
    columns preserves their one-dimensional LHS stratification. This does not
    make independently drawn rows physically feasible sewer networks."""
    u = qmc.LatinHypercube(d=7, seed=seed).random(n)
    if log_uniform:
        return np.exp(np.log(bounds[:, 0]) + u * np.log(bounds[:, 1] / bounds[:, 0]))
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def outputs(x):
    """Return log10 mass predictions as an n-by-4 array (EQ1 to EQ4).

    For each equation, log10(M) = log10(C) + log10(inputs) @ exponents.
    Zero exponents denote absent inputs. Logs of strictly positive values avoid
    fractional-power domain errors. All masses are in kg/day under the fixed,
    rounded SI convention declared above; no input clipping is performed."""
    assert np.all(np.isfinite(x)) and np.all(x > 0)
    return np.log10(COEFF) + np.log10(x) @ ALPHA.T


def uniform_log_variance(a, b):
    """Evaluate Var(log10(X)) analytically for X uniform on [a,b].

    The first two natural-log moments are integrated exactly, then divided by
    log(10)^2. a and b may be arrays, so all seven marginals are handled at once.
    This is a population variance, not an estimate from the synthetic rows."""
    m = (b * np.log(b) - a * np.log(a)) / (b - a) - 1
    m2 = (
        b * (np.log(b) ** 2 - 2 * np.log(b) + 2)
        - a * (np.log(a) ** 2 - 2 * np.log(a) + 2)
    ) / (b - a)
    return (m2 - m * m) / np.log(10) ** 2


def prcc(x, y, k):
    """Compute partial rank correlations for one equation.

    Rank the used input columns and the log output. For each input, regress its
    ranks and the output ranks on all remaining used input ranks plus an
    intercept. Correlate the two residuals. Unused inputs remain NaN. The signs
    are already known analytically for independent inputs; magnitudes describe
    conditional associations under the selected joint design."""
    use = np.flatnonzero(ALPHA[k])
    xr = rankdata(x[:, use], axis=0)
    yr = rankdata(y)
    ans = np.full(7, np.nan)
    for j, v in enumerate(use):
        z = np.column_stack([np.ones(len(x)), np.delete(xr, j, axis=1)])
        rx = xr[:, j] - z @ np.linalg.lstsq(z, xr[:, j], rcond=None)[0]
        ry = yr - z @ np.linalg.lstsq(z, yr, rcond=None)[0]
        ans[v] = np.corrcoef(rx, ry)[0, 1]
    return ans


def diagnostics(x, design):
    """Return log outputs, input diagnostics, pairwise diagnostics and medians.

    The full covariance of log inputs is used for sampled variance allocations.
    These symmetric covariance allocations can be negative and are not Sobol
    indices. Exact independent population shares are calculated separately.
    The upper-tail profile compares input medians in the top 5% and the rest.
    Top-decile overlap is intersection / one decile size, not a Jaccard index.
    All these sampled quantities are conditional on the synthetic input measure."""
    y = outputs(x)
    m = 10**y
    # Population (ddof=0) covariance of the sampled log-input matrix.
    sigma = np.cov(np.log10(x), rowvar=False, ddof=0)
    sens = []
    pairs = []
    for k in range(4):
        p = prcc(x, y[:, k], k)
        den = ALPHA[k] @ sigma @ ALPHA[k]
        # Marginal terms omit covariance; allocations split it symmetrically.
        marginal = ALPHA[k] ** 2 * np.diag(sigma) / den
        alloc = ALPHA[k] * (sigma @ ALPHA[k]) / den
        # Descriptive synthetic upper tail, not a physical extreme-event rate.
        top = y[:, k] >= np.quantile(y[:, k], 0.95)
        # Positive U means larger input medians in the upper 5% output rows.
        profile = np.log2(np.median(x[top], axis=0) / np.median(x[~top], axis=0))
        for v in np.flatnonzero(ALPHA[k]):
            sens.append(
                dict(
                    design=design,
                    equation=k + 1,
                    input=NAMES[v],
                    PRCC=p[v],
                    marginal_variance_fraction=marginal[v],
                    covariance_allocation=alloc[v],
                    U=profile[v],
                    log_variance=den,
                )
            )
    for i in range(4):
        for j in range(i + 1, 4):
            # Stable sorting makes common-tail membership deterministic.
            ti = np.zeros(len(y), dtype=bool)
            tj = ti.copy()
            ti[np.argsort(y[:, i], kind="stable")[-len(y) // 10 :]] = True
            tj[np.argsort(y[:, j], kind="stable")[-len(y) // 10 :]] = True
            pairs.append(
                dict(
                    design=design,
                    i=i + 1,
                    j=j + 1,
                    median_logratio=np.median(y[:, i] - y[:, j]),
                    spearman=spearmanr(y[:, i], y[:, j]).statistic,
                    kendall=kendalltau(y[:, i], y[:, j]).statistic,
                    top_decile_overlap=np.count_nonzero(ti & tj) / np.count_nonzero(ti),
                )
            )
    summary = dict(
        design=design,
        n=len(x),
        **{f"median_EQ{k + 1}": np.median(m[:, k]) for k in range(4)},
        rho_L_A=spearmanr(x[:, 0], x[:, 1]).statistic,
        rho_S_SPD=spearmanr(x[:, 2], x[:, 4]).statistic,
        order_violation=np.mean(x[:, 5] > x[:, 4]),
    )
    return (y, pd.DataFrame(sens), pd.DataFrame(pairs), summary)


def constrained(n, seed, exponential=False, density=False):
    """Retain the first n accepted rows from successive LHS proposal batches.

    Each batch contains n proposals and uses seed + batch_index. O enforces
    SPD4 <= SPD. C additionally imposes the published field-informed L/A range.
    E constructs both slope quantiles from one exponential distribution with
    mean S, then enforces descriptor bounds and the same density envelope.
    The fifth unit-cube coordinate provides p in E; the sixth is unused.
    Accepted samples are NOT LHS samples. These aggregate constraints do not
    certify full hydraulic feasibility or reconstruct deposition-derived PLD."""
    rows = []
    accepted = 0
    proposed = 0
    valid_total = 0
    batch = 0
    while accepted < n:
        x = lhs(n, seed + batch)
        if exponential:
            p = (x[:, 4] - BOUNDS[4, 0]) / (BOUNDS[4, 1] - BOUNDS[4, 0])
            x[:, 4] = -x[:, 2] * np.log1p(-p)
            x[:, 5] = -x[:, 2] * np.log1p(-p / 4)
        ok = x[:, 5] <= x[:, 4]
        if exponential:
            ok &= (
                (x[:, 4] >= BOUNDS[4, 0])
                & (x[:, 4] <= BOUNDS[4, 1])
                & (x[:, 5] >= BOUNDS[5, 0])
                & (x[:, 5] <= BOUNDS[5, 1])
            )
        if density:
            ok &= (x[:, 0] / x[:, 1] >= DENSITY[0]) & (x[:, 0] / x[:, 1] <= DENSITY[1])
        # Preserve proposal order; truncation occurs only after enough accepts.
        selected = x[ok]
        rows.append(selected)
        accepted += len(selected)
        proposed += len(x)
        valid_total += len(selected)
        batch += 1
        if batch > 100:
            raise RuntimeError("Insufficient acceptance")
    return (
        np.vstack(rows)[:n],
        dict(
            seed=seed,
            batches=batch,
            proposed=proposed,
            acceptance=valid_total / proposed,
        ),
    )


def narrowed(indices):
    """Keep the central half of selected intervals' logarithmic widths.

    Their geometric midpoints remain fixed. For ratio r=b/a and midpoint g,
    the narrowed bounds are g/r**0.25 and g*r**0.25. H changes all inputs,
    HL only length, and HP only SPD. The elasticities never change."""
    b = BOUNDS.copy()
    g = np.sqrt(b[:, 0] * b[:, 1])
    r = b[:, 1] / b[:, 0]
    b[indices, 0] = g[indices] / r[indices] ** 0.25
    b[indices, 1] = g[indices] * r[indices] ** 0.25
    return b


def reordering_tables(designs, out):
    """Export large EQ1-versus-EQ4 percentile-rank shifts for B, O, C and E.

    Rows with rank1-rank4 >= 0.5 or <= -0.5 define the two descriptive groups.
    Group input medians and the 100 largest absolute shifts are exported.
    The row column is a zero-based identifier in the corresponding design."""
    groups = []
    largest = []
    for design in ["B", "O", "C", "E"]:
        x = designs[design]
        y = outputs(x)
        r = rankdata(y, axis=0) / len(x)
        delta = r[:, 0] - r[:, 3]
        for group, mask in [("EQ1_up", delta >= 0.5), ("EQ1_down", delta <= -0.5)]:
            row = dict(
                design=design,
                group=group,
                n=int(mask.sum()),
                fraction=mask.mean(),
                median_rank_change=np.median(delta[mask]),
            )
            row.update({name: np.median(x[mask, j]) for j, name in enumerate(NAMES)})
            groups.append(row)
        ix = np.argsort(abs(delta))[-100:][::-1]
        df = pd.DataFrame(
            np.column_stack([ix, x[ix], r[ix, 0], r[ix, 3], delta[ix]]),
            columns=["row"] + NAMES + ["rank_EQ1", "rank_EQ4", "rank_difference"],
        )
        df.insert(0, "design", design)
        largest.append(df)
    pd.DataFrame(groups).to_csv(out / "reordered_groups.csv", index=False)
    pd.concat(largest).to_csv(out / "largest_reorderings.csv", index=False)
