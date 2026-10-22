# Pisano equations: reproducible mathematical audit

Code release prepared on **21 October 2026**, version **1.0.0**.

Companion code for *Assessing the Pisano Equations for Dry-Weather Solids
Deposition in Sewer Systems: Comparative Behaviour and Sensitivity Analysis*.
The manuscript contains the additional methods and checks as Appendices A–E.

## Run the published experiment

Use **Python 3.12**. On Windows, open a terminal in this directory and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

The default run uses the full **N = 200,000** for each design, the fixed seeds
declared in the article, all convergence replications, and all six figures.
It writes the regenerated results to `results/` and compares all 16 CSV tables
with `reference_results/`. A mismatch raises an error; a successful comparison
is recorded in `results/verification_report.json`.

The release was checked using Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3 and Matplotlib 3.10.8. Runtime depends on hardware; the release run
took about 70 seconds in the preparation environment. The dependencies are pinned
in `requirements.txt`; no extra plotting or verification dependency is required.

## Code organisation

All source files use ordinary functions and NumPy arrays; no custom classes or
object-oriented application framework are defined. The standard NumPy, pandas,
SciPy and Matplotlib APIs handle arrays, tables, LHS sampling and plot rendering.
Every scientific function has a docstring describing its role and assumptions.

| File | Purpose |
| --- | --- |
| `reproduce.py` | The entry point; runs every design and export in a fixed order. |
| `model.py` | Units, bounds, four equations, LHS proposals, constraints, analytic and sampled diagnostics. |
| `plots.py` | Paper Figures 1–5 and the Appendix A schematic. |
| `extra_checks.py` | Original four-level flow comparison and US–SI conversion identities. |
| `verify_results.py` | Independent comparison against the stored manuscript reference CSVs. |
| `reference_results/` | The 16 numerical tables used for the article. |
| `results/` | Outputs from the verified release run; the entry script regenerates them. |

Optional exports:

```bash
python reproduce.py --figures-only
python reproduce.py --save-samples
python reproduce.py --out another_results_directory
```

`--figures-only` still samples all 200,000 baseline rows and draws all six figures;
it skips the alternative-design and convergence tables. `--save-samples` additionally
exports eight compressed NPZ matrices. These matrices are omitted from the public
release because the script regenerates them exactly from the declared seeds.
`--out` changes only the output directory, not the experiment.

## Figure map and appearance

| Manuscript | Output stem in `results/` |
| --- | --- |
| Figure 1 | `global_comparison` |
| Figure 2 | `rank_agreement` |
| Figure 3 | `sensitivity` |
| Figure 4 | `eq1_diagnostics` |
| Figure 5 | `eq4_diagnostics` |
| Figure A1 | `Figure_A1_EQ1_descriptors` |

Every figure is exported as PDF, SVG and 600-dpi PNG. Text, axes and contour lines
remain vector elements in PDF/SVG; the continuous colour surface is rasterised
to avoid mesh seams. Figures 4 and 5 retain the blue distribution, orange P50,
red P95 and upper-left legend. The five regenerated main figures were compared
with the manuscript PDFs and matched pixel for pixel at 144 dpi. The release
report is `results/figure_verification_report.json`.

Container metadata such as PDF creation time or SVG element identifiers may
differ between runs. Numerical tables use a relative tolerance of `1e-10` and an
absolute tolerance of `1e-12`; binary file hashes are not a substitute for checking
mathematical and visual reproduction across systems. Matplotlib font rendering
may vary across operating systems even with the same numerical results.

## Equations and input conventions

Every input matrix uses the column order below. Bounds and exponents are stated
explicitly at the top of `model.py`; exponent zero means that an input is absent
from that equation.

| Column | Symbol | Unit |
| --- | --- | --- |
| 0 | L | m |
| 1 | A | ha |
| 2 | S | dimensionless mean slope |
| 3 | D | m |
| 4 | SPD | dimensionless slope quantile at PLD |
| 5 | SPD4 | dimensionless slope quantile at PLD/4 |
| 6 | d | m³ person⁻¹ day⁻¹ |

Outputs are daily deposited-solids mass estimates in kg/day under the historical
clean-pipe assumptions. They are not measured bed stocks or observed event exports.
The fixed SI coefficients are `[0.000264, 0.001091, 0.000369, 0.000709]`.
The audit consistently uses the rounded exponents printed in the article.

`extra_checks.py` transforms the historical US coefficients with that same
exponent convention. This produces an exact unit identity to floating-point
precision while separately documenting the small discrepancy (up to about 0.24%)
from the retained rounded SI coefficients. Changing exponent precision changes
the audited model; it is not merely a unit conversion.

## Designs and interpretation

- B: independent continuous uniform marginals, seed 123.
- O: ordered descriptors, LHS proposal batches starting at seed 223.
- C: ordering and the field-informed L/A envelope, batches starting at seed 323.
- E: common exponential slope quantiles and density restriction, batches starting at seed 423.
- H, HL, HP: narrowed logarithmic widths for all inputs, only L, or only SPD.
- G: independent log-uniform marginals on the original endpoints.
- F: the six baseline non-flow columns plus four equally weighted historical flow levels.

Each rejection batch advances the seed by one. Accepted rows retain proposal
order; the first N are used. Accepted designs O, C and E are not LHS designs.
The precise acceptance accounting is recorded in the manifest.

Independent exact log-variance shares use analytic marginal log variances.
Dependent designs use the full log-input covariance and a symmetric allocation
which may be negative and is not an independent Sobol index. Upper-tail profiles
and top-decile overlaps are descriptive properties of the synthetic experiment.

The baseline is not fitted to occurrence frequencies in real sewers. Selected
constraints do not certify complete hydraulic feasibility. The illustrative
exponential quantile construction does not recover deposition-derived PLD. This
repository contains synthetic computations and published field-context inputs,
not a new observational validation dataset.

## Citation and licence

`CITATION.cff` provides software citation metadata. No repository URL, DOI or
article DOI has been invented. Add the real repository address and archived
version DOI after publication. The included MIT licence is the proposed licence
for the authors' software release; it does not relicense the cited publications
or the journal's LaTeX template, which are not part of this code repository.

The Spanish publishing guide is `GUIA_GITHUB_ZENODO.md`.
