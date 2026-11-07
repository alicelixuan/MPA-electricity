"""Steps 1-2: supply regimes (PELT on carbon-adjusted coal and gas prices, 11 regimes) and piecewise linear supply-curve
fits per regime, giving the hourly slope of the supply curve. From supply_clustering notebooks 05, 06, 08; code unchanged
from the earlier replication package (slopes_plots.py), which reproduces the stored change points and fits exactly.

Randomness: each (regime, min_space) fit has its own generator seeded with [seed, regime, min_space].
The panel (and so the regressions) uses fits with min_space = 0; Figure B1 uses min_space = 1000; Figure 4 uses 0.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import optimize

PENALTY = 30_000.0      # PELT penalty that yields 11 regimes
THRESHOLD = 0.01        # a further breakpoint must lower the SSE by more than 1 %
MAX_BREAKPOINTS = 2
NONNEG_SLOPES = True
MIN_SPACE_PANEL = 0.0
MIN_SPACE_FIG_B = 1000.0
FIG4_REGIMES = (1, 8)


def day_to_date(index_list: list[int]) -> list[pd.Timestamp]:
    """Maps day positions to calendar dates (same helper as ``int_to_day`` in the notebook)."""
    time_index = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
    index_list = list(index_list)
    index_list[-1] = index_list[-1] - 1  # correct last date
    return [time_index[d] for d in index_list]


def day_interval(market: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    seg = market.loc[(market["day"] >= start + 1) & (market["day"] < end + 1)]
    assert len(seg) > 20, f"{start=}, {end=}"
    return seg
# --------------------------------------------------------------------------------------
# Level 2: regime identification (PELT, cause-driven)
# --------------------------------------------------------------------------------------
class _SegmentCost:
    """Sum over series of the squared deviations from the segment mean, in O(1) per segment via prefix sums.

    Reproduces ``calculate_cost`` of cause_driven_approach.py: the cost of (start, end) is computed on
    ``series.iloc[start:end + 1]`` (end inclusive, truncated at the end of the series)."""

    def __init__(self, series: list[np.ndarray]):
        self.n = len(series[0])
        self.s1, self.s2 = [], []
        for x in series:
            xc = np.asarray(x, dtype=float)
            xc = xc - xc.mean()  # SSE is shift invariant; centering keeps the prefix sums accurate
            self.s1.append(np.concatenate([[0.0], np.cumsum(xc)]))
            self.s2.append(np.concatenate([[0.0], np.cumsum(xc**2)]))

    def __call__(self, start: np.ndarray, end: int) -> np.ndarray:
        e = min(end + 1, self.n)
        m = e - start
        total = np.zeros(len(start))
        for s1, s2 in zip(self.s1, self.s2):
            total += (s2[e] - s2[start]) - (s1[e] - s1[start]) ** 2 / m
        return np.maximum(total, 0.0)


def pelt(series: list[np.ndarray], penalty: float, K: float = 0.0) -> list[int]:
    """Pruned exact linear time change point detection (same recursion as ``pelt`` in cause_driven_approach.py).
    Returns the change points including 0 and n."""
    cost = _SegmentCost(series)
    n = cost.n
    F = np.zeros(n + 1)
    F[0] = -penalty
    cp: dict[int, list[int]] = {0: []}
    R = np.array([0])
    for t in range(1, n + 1):
        c = F[R] + cost(R, t) + penalty
        j = int(np.argmin(c))
        F[t] = c[j]
        cp[t] = cp[int(R[j])] + [int(R[j])]
        keep = F[R] + cost(R, t) + K <= F[t]
        R = np.append(R[keep], t)
    return cp[n] + [n]


def identify_regimes(fuel: pd.DataFrame) -> list[int]:
    return pelt([fuel["price_coal"].to_numpy(), fuel["price_gas"].to_numpy()], penalty=PENALTY)

# --------------------------------------------------------------------------------------
# Level 3: piecewise linear fits (random restarts)
# --------------------------------------------------------------------------------------
# Parameter vector (as in piece_wise_linear_algorithm.py): [intercept, x0, dx_1, ..., dx_{k-1}, slope_1, ..., slope_{k+1}]
# with k breakpoints, x_i = x0 + dx_1 + ... + dx_i, intercept = value at x = 0.
def piecewise_linear(x, *params):
    """Continuous piecewise linear function; identical to the original ``piecewise_linear`` up to rounding."""
    x = np.asarray(x, dtype=float)
    k = (len(params) - 2) // 2
    intercept = params[0]
    slopes = params[k + 1 :]
    if k == 0:
        return intercept + slopes[0] * x
    bps = np.cumsum(params[1 : k + 1])
    edges = np.concatenate([[0.0], bps])
    y = intercept + slopes[0] * np.minimum(x, bps[0])
    for j in range(1, k + 1):
        upper = bps[j] if j < k else np.inf
        y = y + slopes[j] * (np.clip(x, edges[j], upper) - edges[j])
    return y


def _bounds(x: np.ndarray, k: int, min_tails: float, min_dist: float):
    if min_tails * 2 + min_dist * (k - 1) > x.max() - x.min():
        raise AssertionError(f"Review input values: {k}, {min_tails} and {min_dist} are incompatible.")
    x_min, x_max = x.min() + min_tails, x.max() - min_tails
    lower, upper = [-np.inf], [np.inf]
    if k > 0:
        lower.append(float(x_min))
        upper.append(float(x_max - min_dist * (k - 1)))
    for _ in range(1, k):
        lower.append(min_dist)
        upper.append(float(x_max - x_min - min_dist * (k - 2)))
    lower += [0.0 if NONNEG_SLOPES else -np.inf] * (k + 1)
    upper += [np.inf] * (k + 1)
    return lower, upper


def _initial_guesses(x, k, n_restarts, min_tails, min_dist, rng: np.random.Generator):
    x_min, x_max = x.min() + min_tails, x.max() - min_tails
    guesses = []
    for _ in range(n_restarts):
        bp_and_dist: list[float] = []
        for i in range(k):
            if i == 0:
                bp_and_dist.append(float(rng.uniform(x_min, x_max - min_dist * (k - 1))))
            else:
                high = x_max - min_dist * (k - i - 1) - sum(bp_and_dist)
                low = min_dist
                assert high > low, f"Something is wrong with bounds, {low}, {high}"
                bp_and_dist.append(float(rng.uniform(low, high)))
        intercept = rng.normal(0, 200)
        slopes = rng.normal(0, 10, k + 1)
        if NONNEG_SLOPES:
            slopes = np.abs(slopes)
        guesses.append([intercept] + bp_and_dist + slopes.tolist())
    return guesses


def fit_piecewise(x, y, min_space: float, n_restarts: int, rng: np.random.Generator) -> dict:
    """Best piecewise linear fit with 0..MAX_BREAKPOINTS breakpoints (``run_piecewise_algorithm`` of the original).

    For each number of breakpoints, ``n_restarts`` least-squares fits start from random guesses and the lowest
    SSE is kept. A model with one more breakpoint is only accepted if it lowers the SSE by more than THRESHOLD."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best_sse, best_params = np.inf, None
    for k in range(0, MAX_BREAKPOINTS + 1):
        dist = min(min_space, (x.max() - x.min() - 2 * min_space) / k) if k != 0 else min_space
        lower, upper = _bounds(x, k, min_space, dist)
        k_sse, k_params = np.inf, None
        for guess in _initial_guesses(x, k, n_restarts, min_space, dist, rng):
            try:
                p, _ = optimize.curve_fit(piecewise_linear, x, y, p0=guess, bounds=(lower, upper))
            except RuntimeError:  # no convergence
                continue
            sse = float(np.sum((piecewise_linear(x, *p) - y) ** 2))
            if sse < k_sse:
                k_sse, k_params = sse, p
        if k_params is None:
            continue
        if k == 0 or abs(k_sse - best_sse) / best_sse > THRESHOLD:
            best_sse, best_params = k_sse, k_params
        else:
            break
    if best_params is None:
        raise RuntimeError("No fit found for the given data and constraints")
    k = (len(best_params) - 2) // 2
    bps = np.cumsum(best_params[1 : k + 1]).tolist() if k else []
    return {
        "n_breakpoints": k,
        "intercept": float(best_params[0]),
        "breakpoints": bps,
        "slopes": [float(s) for s in best_params[k + 1 :]],
        "sse": best_sse,
        "n_obs": len(x),
        "r_squared": 1 - (best_sse / len(x)) / float(np.var(y)),
    }


def regimes_to_index(regimes) -> list[int]:
    return [r - 1 for r in regimes]


def _fit_task(args) -> dict:
    regime, start, end, x, y, min_space, n_restarts, seed = args
    rng = np.random.default_rng([seed, regime, int(min_space)])
    out = fit_piecewise(x, y, min_space, n_restarts, rng)
    return {"regime": regime, "start": start, "end": end, "min_space": min_space, "n_restarts": n_restarts, "seed": seed, **out}


def run_fits(market: pd.DataFrame, change_points: list[int], regimes_by_min_space: dict[float, list[int]], n_restarts: int, seed: int, n_jobs: int = 1) -> pd.DataFrame:
    """Fits every requested regime: ``regimes_by_min_space`` maps min_space -> list of regime numbers (1-based)."""
    tasks = []
    for ms, regimes in regimes_by_min_space.items():
        for i in regimes_to_index(regimes):
            start, end = change_points[i], change_points[i + 1]
            seg = day_interval(market, start, end)
            tasks.append((i + 1, start, end, seg["quantity"].to_numpy(), seg["price"].to_numpy(), float(ms), n_restarts, seed))
    t0 = time.time()
    if n_jobs == 1:
        rows = []
        for k, task in enumerate(tasks, 1):
            rows.append(_fit_task(task))
            print(f"  fit {k}/{len(tasks)} (regime {task[0]}, min_space {task[5]:g}) done [{time.time() - t0:.0f}s]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_fit_task, tasks))
        print(f"  {len(tasks)} fits done [{time.time() - t0:.0f}s]")
    return _fits_to_frame(rows)


def _fits_to_frame(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for r in rows:
        r = dict(r)
        for i, b in enumerate(r.pop("breakpoints")):
            r[f"x{i}"] = b
        for i, s in enumerate(r.pop("slopes"), 1):
            r[f"slope_{i}"] = s
        flat.append(r)
    cols = ["regime", "start", "end", "min_space", "n_obs", "n_breakpoints", "intercept"] + [f"x{i}" for i in range(MAX_BREAKPOINTS)] + [f"slope_{i}" for i in range(1, MAX_BREAKPOINTS + 2)] + ["sse", "r_squared", "n_restarts", "seed"]
    return pd.DataFrame(flat).reindex(columns=cols).sort_values(["min_space", "regime"]).reset_index(drop=True)


def get_fit(fits: pd.DataFrame, regime: int, min_space: float) -> pd.Series:
    row = fits[(fits["regime"] == regime) & (np.isclose(fits["min_space"], min_space))]
    if len(row) != 1:
        raise KeyError(f"No stored fit for regime {regime} and min_space {min_space}; run with --rerun-fits")
    return row.iloc[0]


def fit_params(row: pd.Series) -> tuple[list[float], list[float], float]:
    k = int(row["n_breakpoints"])
    return [float(row[f"x{i}"]) for i in range(k)], [float(row[f"slope_{i}"]) for i in range(1, k + 2)], float(row["intercept"])


def predict(row: pd.Series, x) -> np.ndarray:
    bps, slopes, intercept = fit_params(row)
    dx = [bps[0]] + list(np.diff(bps)) if bps else []
    return piecewise_linear(x, intercept, *dx, *slopes)


# --------------------------------------------------------------------------------------
# Level 4: slopes output (notebook 08) and plots
# --------------------------------------------------------------------------------------
def build_slopes(market: pd.DataFrame, fits: pd.DataFrame, change_points: list[int], min_space: float = MIN_SPACE_FIG_B) -> pd.DataFrame:
    """Hourly marginal slope of the supply curve: the slope of the fitted segment that the hour's quantity falls in
    (``get_specification_slopes`` / ``code_slope`` of notebook 08)."""
    df = market.copy()
    df["slope"] = np.nan
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        bps, slopes, _ = fit_params(get_fit(fits, i + 1, min_space))
        mask = ((df["day"] > start) & (df["day"] <= end)).to_numpy()
        # x < x0 -> slope_1; x0 <= x < x1 -> slope_2; x >= last breakpoint -> last slope
        df.loc[mask, "slope"] = np.asarray(slopes)[np.searchsorted(bps, df.loc[mask, "quantity"].to_numpy(), side="right")]
    return df


def fits_needed(n_regimes: int) -> dict[float, list[int]]:
    """Fits used downstream: all regimes at min_space 0 (panel, Figure 4) and 1000 (Figure B1)."""
    return {MIN_SPACE_PANEL: list(range(1, n_regimes + 1)), MIN_SPACE_FIG_B: list(range(1, n_regimes + 1))}


def hourly_slopes(market: pd.DataFrame, fits: pd.DataFrame, change_points: list[int], min_space: float = MIN_SPACE_PANEL) -> pd.DataFrame:
    """DateTime, slope: the input of the panel."""
    s = build_slopes(market, fits, change_points, min_space)
    return s[["slope"]].rename_axis("DateTime").reset_index()
