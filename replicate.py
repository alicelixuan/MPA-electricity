#!/usr/bin/env python
"""Replication of all figures and tables, from the terminal (replicate.ipynb runs the same with the same settings).

    python replicate.py                                # paper outputs from stored data, about 1 minute
    python replicate.py --rerun-fits --n-jobs 4        # also recompute the supply-curve fits and everything downstream
    python replicate.py --rerun-dispatch-model --n-jobs 8 --solver appsi_highs

Outputs: output/paper/ (always) and output/rerun/<run>/ (reruns, with comparison.md). See README.md.
"""
import argparse

from mpa import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun-regimes", action="store_true", help="step 1, PELT (about 1 s)")
    ap.add_argument("--rerun-fits", action="store_true", help="step 2, supply-curve fits (20-35 min on 2 cores)")
    ap.add_argument("--rerun-dispatch-model", action="store_true", help="step 3, Monte Carlo dispatch (about 230 CPU-hours)")
    ap.add_argument("--rerun-panel", action="store_true", help="step 4, panel (seconds; forced by any rerun above)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--fit-restarts", type=int, default=1000)
    ap.add_argument("--mc-iterations", type=int, default=1000)
    ap.add_argument("--solver", default="glpk", help="glpk (original runs) or appsi_highs")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--run-name", help="folder name under output/rerun/ (default: date and time)")
    ap.add_argument("--dispatch-units", nargs="+", help="dispatch rerun only for these units (EIC); others keep stored results")
    ap.add_argument("--dispatch-start", help="first day of a partial dispatch rerun, YYYY-MM-DD")
    ap.add_argument("--dispatch-end", help="last day of a partial dispatch rerun, YYYY-MM-DD")
    a = ap.parse_args()
    pipeline.run(a.rerun_regimes, a.rerun_fits, a.rerun_dispatch_model, a.rerun_panel, a.seed, a.fit_restarts,
                 a.mc_iterations, a.solver, a.n_jobs, a.run_name, a.dispatch_units, a.dispatch_start, a.dispatch_end)
