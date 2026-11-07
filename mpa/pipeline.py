"""Runs the replication. Always: every figure and table of the paper from the stored data -> output/paper/.
Optionally: rerun slow steps; the rerun step and everything downstream go to
output/rerun/<run>/ together with a comparison with output/paper/. Stored files are never overwritten.

    step 1 regimes (PELT)        stored: data/stored/change_points.json
    step 2 supply-curve fits     stored: data/stored/supply_fits.csv, slopes.parquet
    step 3 dispatch model        stored: data/stored/dispatch_results.parquet
    step 4 panel                 stored: data/stored/analysis_panel.parquet
    step 5 econometrics, step 6 figures and tables: always run
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, econometrics as em, figures, panel as panel_step, supply, tables

FIGURES = ["Figure_3", "Figure_4", "Figure_5", "Figure_6", "Figure_7", "Figure_8", "Figure_B1", "Figure_B2"]


def outputs(out: Path, change_points, fits, panel, dispatch) -> dict[str, pd.DataFrame]:
    """Steps 5-6: all figures and tables from one set of upstream results."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    market, fuel = config.read_market(), config.read_fuel()
    figures.save(figures.fig3(fuel, change_points), out, "Figure_3")
    figures.save(figures.fig4(market, fits, change_points), out, "Figure_4")
    fig, _ = figures.figB1(market, fits, change_points)
    figures.save(fig, out, "Figure_B1")

    df = em.add_variants(panel)
    main = em.regime_switching_logit(df)
    figures.save(figures.fig5(df), out, "Figure_5")
    figures.save(figures.fig6(df), out, "Figure_6")
    figures.save(figures.fig7(df, main), out, "Figure_7")
    for name, key, groups, fig_fn in (
            ("Figure_8", lambda d: d.company, em.companies_by_size(df), figures.fig8),
            ("Figure_B2", lambda d: d.production_type, list(em.TECHS), figures.figB2)):
        coef = figures.coefficients(em.by_group(df, key, groups))
        coef.to_csv(out / f"{name}_data.csv", index=False)
        figures.save(fig_fn(coef), out, name)

    result = tables.all_tables(panel, config.read_load())
    result["Table_A"] = tables.tableA(panel, dispatch, pd.read_csv(config.INPUTS / "unit_parameters.csv"))
    for name, t in result.items():
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"  figures and tables written to {out} [{time.time() - t0:.0f}s]")
    return result


def run(rerun_regimes=False, rerun_fits=False, rerun_dispatch_model=False, rerun_panel=False, seed=2024,
        fit_restarts=1000, mc_iterations=1000, solver="glpk", n_jobs=1, run_name=None, dispatch_units=None,
        dispatch_start=None, dispatch_end=None) -> dict:
    """See the module docstring. dispatch_units / dispatch_start / dispatch_end restrict a dispatch rerun (for tests);
    unit-hours outside it keep the stored results."""
    print("== Paper outputs from stored data")
    paper = outputs(config.PAPER, config.stored_change_points(), config.stored_fits(), config.stored_panel(),
                    config.stored_dispatch())
    if not (rerun_regimes or rerun_fits or rerun_dispatch_model or rerun_panel):
        return {"paper": paper}

    out = config.OUTPUT / "rerun" / (run_name or datetime.now().strftime("%Y-%m-%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    notes = [f"# Rerun {out.name}", "", f"Settings: regimes={rerun_regimes}, fits={rerun_fits}, dispatch={rerun_dispatch_model}, "
             f"panel={rerun_panel}, seed={seed}, fit_restarts={fit_restarts}, mc_iterations={mc_iterations}, solver={solver}", ""]
    market = config.read_market()

    # step 1
    change_points = config.stored_change_points()
    if rerun_regimes:
        print("== Step 1: regimes (PELT)")
        new = supply.identify_regimes(config.read_fuel())
        (out / "change_points.json").write_text(json.dumps({"change_points": new}))
        same = new == change_points
        notes.append(f"Change points: {'identical to stored' if same else f'differ from stored {change_points}: {new}'}")
        if not same:
            rerun_fits = True                      # stored fits belong to the stored regimes
        change_points = new

    # step 2
    fits, slopes = config.stored_fits(), config.stored_slopes()
    if rerun_fits:
        print(f"== Step 2: supply-curve fits ({fit_restarts} restarts, seed {seed})")
        fits = supply.run_fits(market, change_points, supply.fits_needed(len(change_points) - 1), fit_restarts, seed, n_jobs)
        fits.to_csv(out / "supply_fits.csv", index=False)
        new = supply.hourly_slopes(market, fits, change_points)
        new.to_parquet(out / "slopes.parquet", index=False)
        c = slopes.merge(new, on="DateTime", suffixes=("_stored", "_rerun"))
        d = (c.slope_stored - c.slope_rerun).abs()
        notes.append(f"Hourly slopes: {100 * (d < 1e-4).mean():.2f} % of hours within 1e-4 of stored, correlation "
                     f"{c.slope_stored.corr(c.slope_rerun):.4f}")
        slopes = new

    # step 3
    dispatch = config.stored_dispatch()
    if rerun_dispatch_model:
        print(f"== Step 3: dispatch model ({mc_iterations} iterations, solver {solver}, seed {seed})")
        from .dispatch_model import run_dispatch_model
        new = run_dispatch_model(mc_iterations, seed, solver, n_jobs, dispatch_units, dispatch_start, dispatch_end)
        new.to_parquet(out / "dispatch_results.parquet", index=False)
        c = new.merge(dispatch, on=["EIC", "DateTime"], suffixes=("_rerun", "_stored"))
        for v in ("state", "dispatch", "startup", "startup_cost", "marginal_cost"):
            notes.append(f"Dispatch results, {v}: mean {c[v + '_rerun'].mean():.4f} (stored {c[v + '_stored'].mean():.4f}), "
                         f"correlation {c[v + '_rerun'].corr(c[v + '_stored']):.3f}, {len(c):,} unit-hours")
        key = ["EIC", "DateTime"]
        dispatch = pd.concat([dispatch.set_index(key).drop(new.set_index(key).index, errors="ignore"),
                              new.set_index(key)]).reset_index()

    # step 4
    panel = config.stored_panel()
    if rerun_regimes or rerun_fits or rerun_dispatch_model or rerun_panel:
        print("== Step 4: panel")
        new = panel_step.build_panel(dispatch, slopes)
        new.to_parquet(out / "analysis_panel.parquet", compression="zstd")
        num = [c for c in new.columns if new[c].dtype.kind == "f"]
        same = np.allclose(new[num].to_numpy(float), panel[num].to_numpy(float), equal_nan=True, atol=1e-9)
        notes.append(f"Panel: {'identical to stored' if same else 'differs from stored'}")
        panel = new

    # steps 5-6
    print("== Steps 5-6: figures and tables from the rerun results")
    rerun = outputs(out, change_points, fits, panel, dispatch)
    notes += ["", compare_runs(paper, rerun)]
    (out / "comparison.md").write_text("\n".join(notes))
    print(f"  comparison written to {out / 'comparison.md'}")
    return {"paper": paper, "rerun": rerun, "rerun_dir": out}


def compare_runs(paper: dict, rerun: dict) -> str:
    """Every table value: paper output vs rerun output."""
    rows = []
    for name, t in paper.items():
        m = t.merge(rerun[name], on=["row", "column"], suffixes=("_paper", "_rerun"))
        m.insert(0, "table", name)
        rows.append(m)
    m = pd.concat(rows)
    m["diff"] = m.value_rerun - m.value_paper
    changed = m[(m["diff"].abs() > 1e-9) | (m.stars_paper != m.stars_rerun)]
    lines = [f"## Tables: {len(changed)} of {len(m)} values differ from output/paper", ""]
    if len(changed):
        lines.append("| Table | Row | Column | Paper | Rerun |\n|---|---|---|---|---|")
        for r in changed.itertuples():
            lines.append(f"| {r.table} | {r.row} | {r.column} | {r.value_paper:.4g}{r.stars_paper} | {r.value_rerun:.4g}{r.stars_rerun} |")
    return "\n".join(lines)
