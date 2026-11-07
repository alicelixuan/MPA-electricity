"""Monte Carlo unit-commitment model behind the dispatch results (competitive benchmark, one unit at a time).

Port of ML-Strom/code/optimization_clean.py (same model, same parameters), reading its inputs from this package
instead of the raw ML-Strom pickles, with a seed and optional parallelisation.

For each unit and each Monte Carlo iteration, parameters are drawn (efficiency +-10 %, cold-start factor +-10 %,
fuel-cost factor +-5 %; normal with std = percent / 200) and the unit's profit is maximised over rolling
30-day horizons (+7 days buffer that is discarded), subject to
    dispatch <= capacity * on,  min_dispatch * on <= dispatch,  on <= dispatch,  startup[t] >= on[t] - on[t-1].
Objective: sum_t (price - marginal_cost) * dispatch - startup_cost * startup.
Results are averaged over the iterations.

Inputs: unit-hours and price (data/stored/analysis_panel.parquet), data/inputs/fuel_costs_hourly.csv.gz (carbon-adjusted
fuel costs, EUR/MWh thermal), data/inputs/unit_parameters.csv (technology, nameplate capacity, efficiency).
Output: the format of data/stored/dispatch_results.parquet (written by the pipeline to output/rerun/<run>/).

Needs: pyomo and a MILP solver. GLPK (the solver of the original runs): `conda install -c conda-forge glpk`
or `apt install glpk-utils`; alternatively HiGHS: `pip install highspy` and `--solver appsi_highs`.

The result is random (Monte Carlo): other seeds, solvers or numbers of iterations give slightly different values.
Runtime is hours for all 40 units and 1000 iterations: use MC_ITERATIONS and N_JOBS (or --mc-iterations, --n-jobs).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HORIZON_DAYS = 30
BUFFER_DAYS = 7
SEED = 2024
ITERATIONS = 1000   # as in the stored results

FUEL_COL = {  # fuel series [EUR/MWh thermal] by technology
    "Fossil Brown coal/Lignite": "Clean Lignite [EUR/MWh]",
    "Fossil Hard coal": "Clean Coal [EUR/MWh]",
    "Fossil Gas Combined cycle": "Clean Gas [EUR/MWh]",
    "Fossil Gas gas/steam turbine": "Clean Gas [EUR/MWh]",
}
MIN_SHARE = {"lignite": 0.40, "hard_coal": 0.40, "ccgt": 0.45, "ocgt": 0.20}
TECH = {"Fossil Brown coal/Lignite": "lignite", "Fossil Hard coal": "hard_coal",
        "Fossil Gas Combined cycle": "ccgt", "Fossil Gas gas/steam turbine": "ocgt"}
DEPRECIATION = {"big_coal": 49, "small_coal": 105, "ccgt": 60, "ocgt": 24}          # EUR / MW
COLD_START_FUEL = {"big_coal": 5.9, "small_coal": 2.7, "ccgt": 2.8, "ocgt": 0.1}    # MWh_thermal / MW
COLD_START_FACTOR = {"big_coal": 0.5, "small_coal": 0.5, "ccgt": 1.0, "ocgt": 1.0}


def sample_plus_minus_percent(rng, number: float, percent: float = 10.0) -> float:
    return number * (1 + rng.normal(0, percent / 200))


def startup_cost_series(tech: str, capacity: float, fuel: np.ndarray, fuel_factor: float, startup_factor: float):
    """Startup cost [EUR] for every hour (get_startup_costs of the original)."""
    cls = tech
    if tech in ("lignite", "hard_coal"):
        cls = "big_coal" if capacity > 500 else "small_coal"
    return (DEPRECIATION[cls] * capacity
            + COLD_START_FUEL[cls] * capacity * (fuel * fuel_factor) * COLD_START_FACTOR[cls] * startup_factor)


def write_model(pyo, price, marginal_cost, startup_cost, capacity, min_dispatch):
    """The unit's MILP for one planning horizon (write_model of the original)."""
    T = len(price)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(T)
    m.price = pyo.Param(m.T, initialize={i + 1: float(v) for i, v in enumerate(price)})
    m.marginal_cost = pyo.Param(m.T, initialize={i + 1: float(v) for i, v in enumerate(marginal_cost)})
    m.startup_cost = pyo.Param(m.T, initialize={i + 1: float(v) for i, v in enumerate(startup_cost)})
    m.capacity = capacity
    m.min_dispatch = min_dispatch

    m.dispatch = pyo.Var(m.T, bounds=(0, capacity))
    m.on = pyo.Var(m.T, within=pyo.Binary)
    m.startup = pyo.Var(m.T, within=pyo.Binary)

    m.profit = pyo.Objective(
        rule=lambda m: sum((m.price[t] - m.marginal_cost[t]) * m.dispatch[t] - m.startup_cost[t] * m.startup[t]
                           for t in m.T), sense=pyo.maximize)
    m.dispatch_constraint = pyo.Constraint(m.T, rule=lambda m, t: m.dispatch[t] <= m.capacity * m.on[t])
    m.minimum_dispatch = pyo.Constraint(m.T, rule=lambda m, t: m.min_dispatch * m.on[t] <= m.dispatch[t])
    m.on_off_constraint = pyo.Constraint(m.T, rule=lambda m, t: m.on[t] <= m.dispatch[t])
    m.startup_constraint = pyo.ConstraintList()
    for t in m.T:
        if t > 1:
            m.startup_constraint.add(m.startup[t] >= m.on[t] - m.on[t - 1])
    return m


def solve_unit(price, fuel, tech, capacity, efficiency, rng, solver):
    """One Monte Carlo draw for one unit: sample parameters, solve all planning horizons.
    Returns arrays (marginal_cost, status, generation, startup, startup_cost) of length len(price)."""
    import pyomo.environ as pyo

    eff = sample_plus_minus_percent(rng, efficiency, 10)
    startup_factor = sample_plus_minus_percent(rng, 1, 10)
    fuel_factor = sample_plus_minus_percent(rng, 1, 5)
    mc = fuel * fuel_factor / eff
    sc = startup_cost_series(tech, capacity, fuel, fuel_factor, startup_factor)
    min_dispatch = MIN_SHARE[tech] * capacity

    H, B = 24 * HORIZON_DAYS, 24 * BUFFER_DAYS
    n = len(price)
    status, gen, start = np.zeros(n), np.zeros(n), np.zeros(n)
    opt = pyo.SolverFactory(solver)
    for m_ in range(int(np.ceil(n / H))):
        lo, hi = m_ * H, min((m_ + 1) * H + B, n)
        model = write_model(pyo, price[lo:hi], mc[lo:hi], sc[lo:hi], capacity, min_dispatch)
        opt.solve(model)
        keep = min(H, hi - lo)
        for t in range(1, keep + 1):
            status[lo + t - 1] = pyo.value(model.on[t])
            gen[lo + t - 1] = pyo.value(model.dispatch[t])
            start[lo + t - 1] = pyo.value(model.startup[t])
    return mc, status, gen, start, sc


def _task(args):
    (eic, iteration, price, fuel, tech, capacity, efficiency, seed, uidx, solver) = args
    rng = np.random.default_rng([seed, iteration, uidx])
    return eic, solve_unit(price, fuel, tech, capacity, efficiency, rng, solver)


def run_dispatch_model(iterations=ITERATIONS, seed=SEED, solver="glpk", n_jobs=1, eics=None, start=None, end=None) -> pd.DataFrame:
    """Runs the model for the units and hours of the stored panel; returns EIC, DateTime, state, dispatch,
    marginal_cost, startup, startup_cost (averages over iterations), the format of data/stored/dispatch_results."""
    from . import config

    pub = config.stored_panel().reset_index()[["EIC", "DateTime", "price"]].sort_values(["EIC", "DateTime"])
    if eics is not None:
        pub = pub[pub.EIC.isin(eics)]
    if start is not None:
        pub = pub[pub.DateTime >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        pub = pub[pub.DateTime < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    fuel_df = pd.read_csv(config.INPUTS / "fuel_costs_hourly.csv.gz")
    fuel_df["DateTime"] = pd.to_datetime(fuel_df["DateTime"], utc=True)
    par = pd.read_csv(config.INPUTS / "unit_parameters.csv").set_index("EIC")
    pub = pub.merge(fuel_df, on="DateTime", how="inner")

    units = list(pub.EIC.unique())
    per_unit = {}
    for u in units:
        d = pub[pub.EIC == u]
        p = par.loc[u]
        per_unit[u] = dict(df=d, price=d.price.to_numpy(float), fuel=d[FUEL_COL[p.production_type]].to_numpy(float),
                           tech=TECH[p.production_type], cap=float(p.installed_capacity_model), eff=float(p.efficiency))
    print(f"dispatch model: {len(units)} units, {len(pub):,} unit-hours, {iterations} iterations, solver {solver}, seed {seed}")

    acc = {u: np.zeros((5, len(per_unit[u]["price"]))) for u in units}
    tasks = [(u, k, per_unit[u]["price"], per_unit[u]["fuel"], per_unit[u]["tech"], per_unit[u]["cap"],
              per_unit[u]["eff"], seed, i, solver)
             for k in range(iterations) for i, u in enumerate(units)]
    t0 = time.time()
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = ex.map(_task, tasks, chunksize=1)
            for j, (u, res) in enumerate(results):
                acc[u] += np.vstack(res)
                if (j + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  {j + 1}/{len(tasks)} unit-iterations, {time.time() - t0:.0f}s", flush=True)
    else:
        for j, t in enumerate(tasks):
            u, res = _task(t)
            acc[u] += np.vstack(res)
            if (j + 1) % max(1, len(tasks) // 20) == 0:
                print(f"  {j + 1}/{len(tasks)} unit-iterations, {time.time() - t0:.0f}s", flush=True)

    frames = []
    for u in units:
        mc, st, gen, su, sc = acc[u] / iterations
        frames.append(pd.DataFrame({"EIC": u, "DateTime": per_unit[u]["df"].DateTime.to_numpy(), "state": st,
                                    "dispatch": gen.clip(min=0), "marginal_cost": mc, "startup": su, "startup_cost": sc}))
    return pd.concat(frames, ignore_index=True)
