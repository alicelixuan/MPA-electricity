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

The run takes about one minute (tested with Python 3.14.6 on Linux). It creates `output/paper/` with every figure
(`.png`, `.svg`) and table (`.csv`) of the paper, computed from the stored data, and shows them in the notebook.

## Steps and reruns

`replicate.ipynb` and `replicate.py` call `mpa/pipeline.py`, which runs six steps. By default steps 1–4 are read from
`data/stored/`. A rerun flag recomputes that step and every step after it.

| Step | Code | Stored result (`data/stored/`) | Rerun flag | Rerun time |
|---|---|---|---|---|
| 1. Supply regimes (PELT, 11 regimes) | `supply.py` | `change_points.json` | `RERUN_REGIMES` | seconds |
| 2. Supply-curve fits, hourly slopes | `supply.py` | `supply_fits.csv`, `slopes.parquet` | `RERUN_FITS` | 20–35 min on 2 cores |
| 3. Monte Carlo dispatch model | `dispatch_model.py` | `dispatch_results.parquet` | `RERUN_DISPATCH_MODEL` | ~13 CPU-s per unit and iteration: ~150 CPU-hours for 1000 iterations |
| 4. Analysis panel | `panel.py` | `analysis_panel.parquet` | `RERUN_PANEL` | seconds |
| 5. Logit models | `econometrics.py` | – | always runs | – |
| 6. Figures and tables | `figures.py`, `tables.py` | – | always runs | ~1 min |

Set the flags in the notebook's first cell.

* **Where reruns go.** A rerun creates `output/rerun/<run>/`, which holds the recomputed intermediate results, all
  figures and tables, and `comparison.md` (rerun vs `output/paper/`). `output/paper/` and `data/` are never
  overwritten.
* **Dispatch model.** A rerun needs `pip install -r requirements-model.txt` and a solver.
  * Solver: GLPK, as in the original runs (`apt install glpk-utils` or `conda install -c conda-forge glpk`), or HiGHS
    (`SOLVER = "appsi_highs"`).
  * Speed and size: `N_JOBS` sets the number of parallel processes. `DISPATCH_UNITS` reruns only some units; the other
    unit-hours keep the stored results.
* **Randomness.** The fits and the dispatch model are seeded (`SEED = 2024`), so a rerun is repeatable. The stored
  results come from unseeded original runs, so a rerun gets close to them but does not match them exactly.
  * Refitted hourly slopes agree with the paper's within 1e-4 in 98 % of hours; all R² agree to four decimals.
  * A 10-iteration dispatch rerun of all 40 units correlates 0.99 with the stored 1000-iteration results (state and
    dispatch).

## Data

All files in `data/` may be redistributed. The paper documents how they were compiled.

| File (`data/inputs/`) | Content | Source |
|---|---|---|
| `market_hourly.csv` | Hourly residual load (MW) and day-ahead price (EUR/MWh) | ENTSO-E |
| `load_hourly.csv` | Hourly load (MW) | ENTSO-E |
| `units_observed.parquet` | Hourly generation, available and installed capacity, owner and technology of every unit of the sample companies | ENTSO-E, EEX, Fusar Bassini (2025) |
| `chp_generation.parquet` | Hourly generation of each company's CHP units | as above |
| `unit_parameters.csv` | Technology, nameplate capacity, efficiency per unit | EEX, OPSD |
| `unit_control_area.csv` | Control area of each unit, from its location and network operator | EEX, ENTSO-E |
| `redispatch_entsoe.parquet` | Redispatch actions in German control areas (13.1.A) | ENTSO-E |
| `fuel_prices_daily.csv` | Carbon-adjusted gas and coal prices (EUR/MWh thermal) | Investing.com, ICAP, UBA |
| `fuel_costs_hourly.csv.gz` | Hourly carbon-adjusted coal, lignite and gas costs | Investing.com, ICAP, UBA |

Sources:

* ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
* EEX Transparency Platform: https://www.eex-transparency.com/power
* Fusar Bassini (2025): doi:10.5281/zenodo.15204429
* OPSD: https://open-power-system-data.org/
* ICAP EU ETS carbon price: https://icapcarbonaction.com/en/ets-prices
* Investing.com:
  * coal CIF ARA: https://www.investing.com/commodities/coal-(api2)-cif-ara-futures-historical-data
  * gas TTF: https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data
* Umweltbundesamt (UBA), carbon intensity of fuels:
  https://www.umweltbundesamt.de/sites/default/files/medien/11850/publikationen/23_2024_cc_strommix_11_2024.pdf

## Model variables

Unit *i*, company *j*, hour *h*, month *m*:

* **Contribution margin:** $m_{i,h} = p_h - c^{var}_{i,h}$, where $c^{var}_{i,h}$ is the variable cost from the
  dispatch model, averaged over its iterations:
  * $c^{var}_{i,h} = (c^{fuel}_h + c^{carbon}_h)/\sigma_i$, with $\sigma_i$ the thermal conversion efficiency;
  * $c^{carbon}_h$ is the carbon price times 0.201 (gas) or 0.338 (coal) t CO2 per MWh thermal.
* **Exposure:**
  * company generation $G_{j,h} = \sum_{i \in j} G_{i,h}$;
  * hedged quantity $Q^{hedged}_{j,m}$ = $r$ × the mean of $G_{j,h}$ over month *m*, separately for peak and off-peak
    hours (hedge rate $r = 1$ in the main model);
  * net exposure $E_{j,h} = G_{j,h} - Q^{hedged}_{j,m}$.
* **Net profit per MW**, with $\delta_h$ the slope of the supply curve:
  * withholding: $\pi^{W}_{i,j,h} = \delta_h E_{j,h} - m_{i,h}$;
  * push-in: $\pi^{P}_{i,j,h} = -\delta_h E_{j,h} + m_{i,h}$.
* **Competitive benchmark:** $z_{i,h}$ = 1 if the Monte Carlo average of the model state is ≥ 0.95, 0 if ≤ 0.05.
* **Deviation in dispatch:** $y_{i,h}$ = observed status (generation > 0) − $z_{i,h}$. A value of −1 is set to 0 when
  the unit is fully unavailable.

## Requirements

`requirements.txt` for the replication; `requirements-model.txt` only for a dispatch model rerun.

## License

The code in this repository is MIT licensed, see `LICENSE.txt`.
