# Market power abuse in wholesale electricity markets

The code and data in this repository replicate the analysis in

> Xu, A. L., Sánchez Canales, J., Fusar Bassini, C., Kaack, L. H., Hirth, L., 2025: Market power abuse in wholesale
> electricity markets.

## Quick start

```bash
python3.14 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook replicate.ipynb                            # Run All; or: python replicate.py
```

The default run takes about one minute. It produces every figure and table of the paper from the stored data in
`output/paper/` and shows them in the notebook. Tested with Python 3.14.6 on Linux.

## Which code produces which result

All code is in `mpa/`. `replicate.ipynb` and `replicate.py` call `mpa/pipeline.py`, which runs the steps below.

| Paper | Output file (`output/paper/`) | Code |
|---|---|---|
| Figures 1, 2 | – (illustrations, not produced from data) | – |
| Figure 3 | `Figure_3.png/.svg` | `figures.fig3` |
| Figure 4 | `Figure_4` | `figures.fig4` |
| Figure 5 | `Figure_5` | `figures.fig5` |
| Figure 6 | `Figure_6` | `figures.fig6` |
| Figure 7 | `Figure_7` | `figures.fig7` |
| Figure 8 | `Figure_8`, `Figure_8_data.csv` | `figures.fig8` |
| Figure B1 | `Figure_B1` | `figures.figB1` |
| Figure B2 | `Figure_B2`, `Figure_B2_data.csv` | `figures.figB2` |
| Table 1 (and shares of profitable unit-hours, Section 5.3) | `Table_1.csv` | `tables.table1` |
| Tables 2–7 | `Table_2.csv` … `Table_7.csv` | `tables.table2` … `tables.table7` |
| Table in Appendix A | `Table_A.csv` | `tables.tableA` |
| Tables B1–B3 | `Table_B1.csv` … `Table_B3.csv` | `tables.tableB1` … `tables.tableB3` |

## Steps, stored results and reruns

| Step | Code | Stored result (`data/stored/`) | Rerun flag | Time | Random |
|---|---|---|---|---|---|
| 1. Supply regimes (PELT, 11 regimes) | `supply.py` | `change_points.json` | `RERUN_REGIMES` | ~1 s | no |
| 2. Piecewise linear supply-curve fits, hourly slopes | `supply.py` | `supply_fits.csv`, `slopes.parquet` | `RERUN_FITS` | 20–35 min (2 cores) | seeded |
| 3. Monte Carlo unit-commitment dispatch model | `dispatch_model.py` | `dispatch_results.parquet` | `RERUN_DISPATCH_MODEL` | ~230 CPU-hours | seeded |
| 4. Analysis panel (deviation, net exposure, net profit, redispatch) | `panel.py` | `analysis_panel.parquet` | `RERUN_PANEL` | seconds | no |
| 5. Logit models | `econometrics.py` | – | always | – | no |
| 6. Figures and tables | `figures.py`, `tables.py` | – | always | ~1 min | no |

By default every step reads the stored result of the step before. A rerun flag recomputes that step and every step
downstream; results go to `output/rerun/<run>/`, with `comparison.md` comparing them with `output/paper/`. Stored files
are never overwritten. Set the flags in the notebook's first cell, or use `python replicate.py --help`.

* **Dispatch model rerun** needs `pip install -r requirements-model.txt` and a MILP solver: GLPK, as in the original runs
  (`apt install glpk-utils` or `conda install -c conda-forge glpk`), or HiGHS (`SOLVER = "appsi_highs"`). `N_JOBS` splits
  it over cores. `DISPATCH_UNITS` (or `--dispatch-units`, `--dispatch-start`, `--dispatch-end`) runs a subset; the
  other unit-hours keep the stored results.
* **Randomness.** The fits and the dispatch model start from random draws. They are seeded (`SEED = 2024`, one generator
  per fit and per iteration and unit), so a rerun is repeatable. The stored results come from unseeded original runs,
  so a rerun approximates them rather than reproducing them:
  * Fits with 1000 restarts and the pinned versions keep every fit's number of breakpoints and every R² to four
    decimals. The Figure B1 fits (min_space = 1000) move less than 6 MW.
  * Two breakpoints of the panel fits (min_space = 0) sit on a flat error surface and move: regime 3 by 637 MW,
    regime 1 by 164 MW. The relative change in SSE is at most 1e-5.
  * Hourly slopes from a refit agree with the slopes used in the paper (`slopes.parquet`) within 1e-4 in 98.3 % of
    hours (correlation 0.996). The original runs were unseeded with 100–200 restarts, and the stored fits themselves
    agree with the paper's slopes in 99.2 % of hours.
  * A 10-iteration dispatch run on two units correlates 0.98 (state) and 0.99 (dispatch) with the stored 1000-iteration
    results.
* **Panel.** Rebuilt from the stored results of steps 2–3, the panel equals `analysis_panel.parquet` exactly; tables
  and figures from such a rerun equal `output/paper/`.

## How table values are computed

The rules below are applied to every cell of the table concerned (`tables.py`):

* **Tables 2 and 4–7.** Stars mark p ≤ 0.01 (\*\*\*) and p ≤ 0.05 (\*\*). The 10 % star (\*) uses p rounded to one
  decimal.
* **Tables 3 and B1, occurrence.** Occurrence is the share of the model sample at or above the scenario's printed net
  profit. The threshold is capped at the sample maximum, because row iv is built from rounded parameters to match the
  largest observation.
* **Table B2.** Volume shares are averages of hourly shares (expected MW ÷ load in that hour).
* **Table B3.** Negative deviation is 100 minus the other two rounded shares, so each row sums to 100.

## Data

### Primary data sources

* Generation, capacity and market data, including the wholesale electricity price, load and unit generation:
  ENTSO-E Transparency Platform (https://transparency.entsoe.eu/).
* Outage reports, unit and company information: EEX Transparency Platform (https://www.eex-transparency.com/power);
  Fusar Bassini (2025), doi:10.5281/zenodo.15204429.
* Capacity and thermal conversion efficiency estimates, additionally: OPSD (https://open-power-system-data.org/).
* EU ETS carbon price: ICAP (https://icapcarbonaction.com/en/ets-prices).
* Coal CIF ARA price index: Investing.com (https://www.investing.com/commodities/coal-(api2)-cif-ara-futures-historical-data).
* Gas TTF price index: Investing.com (https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data).
* Carbon intensity of fuels, t CO2 per MWh primary energy: Umweltbundesamt
  (https://www.umweltbundesamt.de/sites/default/files/medien/11850/publikationen/23_2024_cc_strommix_11_2024.pdf).
* Redispatch volumes (ENTSO-E 13.1.A): ENTSO-E Transparency Platform.

The paper documents how the input data were compiled from these sources. All files in `data/` may be redistributed.

### Input files

| File (`data/inputs/`) | Content |
|---|---|
| `fuel_prices_daily.csv` | Carbon-adjusted gas and coal prices, EUR/MWh thermal, 2019–2024 (regimes) |
| `market_hourly.csv` | Hourly residual load (MW) and day-ahead price (EUR/MWh) |
| `fuel_costs_hourly.csv.gz` | Hourly carbon-adjusted coal, lignite and gas costs (dispatch model) |
| `unit_parameters.csv` | Technology, nameplate capacity and efficiency per unit (dispatch model) |
| `units_observed.parquet` | Hourly generation, available and installed capacity, owner and technology of every unit of the sample companies |
| `chp_generation.parquet` | Hourly generation of each company's CHP units |
| `redispatch_entsoe.parquet` | Redispatch actions in German control areas |
| `unit_control_area.csv` | Control area of each unit |
| `load_hourly.csv` | Hourly load (MW) |

### Model variables (`mpa/panel.py`, `mpa/econometrics.py`)

Unit *i*, company *j*, hour *h*, month *m*:

* Contribution margin: $m_{i,h} = p_h - c^{var}_{i,h}$, with variable cost
  $c^{var}_{i,h} = (c^{Gas|Coal}_h + c^{Carbon}_h)/\sigma_i$; $c^{Carbon}_h = Carbon_h \cdot 0.201$ for gas-fired
  and $Carbon_h \cdot 0.338$ for coal-fired units; $\sigma_i$ = thermal conversion efficiency.
* Company generation $G_{j,h} = \sum_{i \in j} G_{i,h}$, hedged quantity
  $Q^{hedged}_{j,m} = r \cdot \frac{1}{H}\sum_{h \in H} G_{j,h}$ (separately for peak and off-peak hours of month
  *m*; hedge rate *r* = 1 in the main model) and net exposure $E_{j,h} = G_{j,h} - Q^{hedged}_{j,m}$.
* Net profit from withholding 1 MW: $\pi^{W}_{i,j,h} = \delta_h E_{j,h} - m_{i,h}$, with $\delta_h$ the slope of the
  supply curve.
* Net profit from pushing in 1 MW: $\pi^{P}_{i,j,h} = -\delta_h E_{j,h} + m_{i,h}$.
* Competitive dispatch status $z_{i,h}$: 1 if the Monte Carlo average of the model state is ≥ 0.95, 0 if ≤ 0.05.
  Deviation in dispatch $y_{i,h}$ = observed status (generation > 0) − $z_{i,h}$; −1 is not counted as withholding when
  the unit is fully unavailable.

## Requirements

* `requirements.txt`: numpy, pandas, scipy, statsmodels, matplotlib, pyarrow, and Jupyter for the notebook. Versions
  are pinned: Table 5 uses Nelder–Mead estimation as in the paper, and its push-in column can depend on the
  scipy/statsmodels versions.
* `requirements-model.txt`: only for the dispatch model rerun.
* Disk: about 100 MB of data.

## License

The code in this repository is MIT licensed, see `LICENSE.txt`.
