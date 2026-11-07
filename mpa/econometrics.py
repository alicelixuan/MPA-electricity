"""Step 5: regime-switching logit (Section 4.3) and its variants. Ported from Analysis/Econometric analysis.ipynb
and ML-Strom notebook 8; one model function for all tables and figures.

Model 1 (z = 1, competitive benchmark says run):    P(y = -1) = logistic(b0 + b1 * net profit of withholding)
Model 2 (z = 0, competitive benchmark says idle):   P(y =  1) = logistic(b0 + b1 * net profit of push-in)
Standard errors clustered by unit (EIC). Estimation as in the paper: Nelder-Mead, at most 1000 iterations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

Z, Y = "competitive_dispatch_status_discretized", "deviation_discretized"
WITHHOLD, PUSHIN = "net_profit_withhold_marginal", "net_profit_pushin_marginal"
TECHS = {"Fossil Brown coal/Lignite": "Lignite", "Fossil Hard coal": "Hard coal",
         "Fossil Gas Combined cycle": "CCGT", "Fossil Gas gas/steam turbine": "Gas other types"}
HEDGE_RATES = (1.0, 0.7, 0.0)


def add_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Net profit for other hedge rates (Table 6) and for 1/3-unit deviation (Table 7), from the panel columns.
    Hedge rate r: E_r = company generation - r * hedged quantity. 1/3 unit, k = installed capacity / 3:
    withholding k * (slope * (E + k) - margin) (notebook 8, cell 3); push-in k * (-slope * E + margin), the version
    that reproduces Table 7 (notebook 8 has -slope * (E - actual generation))."""
    df = df.copy()
    gen, hedged = df.company_gen.astype(float), df.hedged_quantity.astype(float)
    for r in HEDGE_RATES[1:]:
        e = gen - r * hedged
        df[f"{WITHHOLD}_hedge_{r:g}"] = df.slope * e - df.margin
        df[f"{PUSHIN}_hedge_{r:g}"] = -df.slope * e + df.margin
    k, e = df.installed_capacity / 3, df.net_exposure
    df["net_profit_withhold_third_unit"] = k * (df.slope * (e + k) - df.margin)
    df["net_profit_pushin_third_unit"] = k * (-df.slope * e + df.margin)
    return df


def logit(y: pd.Series, x: pd.DataFrame, cluster: bool = True, method: str = "nm"):
    groups = x.index.get_level_values("EIC")
    kw = dict(cov_type="cluster", cov_kwds={"groups": groups}) if cluster else {}
    return sm.Logit(y.astype(int), sm.add_constant(x)).fit(maxiter=1000, method=method, disp=False, **kw)


def regime_switching_logit(df: pd.DataFrame, withhold=(WITHHOLD,), pushin=(PUSHIN,), cluster=True, method="nm"):
    """Returns (model 1: withholding, model 2: push-in)."""
    d1, d2 = df[df[Z] == 1], df[df[Z] == 0]
    m1 = logit(d1[Y] == -1, d1[list(withhold)], cluster, method)
    m2 = logit(d2[Y] == 1, d2[list(pushin)], cluster, method)
    return m1, m2


def by_group(df: pd.DataFrame, key, groups) -> dict:
    """Model per group (company, technology, year). Groups where clustering fails (a single unit) are
    estimated without clustering, as in the original notebook."""
    out = {}
    for g in groups:
        d = df[key(df) == g]
        try:
            out[g] = regime_switching_logit(d)
        except Exception:
            out[g] = regime_switching_logit(d, cluster=False)
    return out


def companies_by_size(df: pd.DataFrame) -> list[str]:
    """Companies ranked by their largest absolute net exposure (order of Figure 8)."""
    size = df.groupby("company").net_exposure.agg(lambda s: s.abs().max())
    return size.sort_values(ascending=False).index.tolist()


def predict(result, net_profit) -> np.ndarray:
    b0, b1 = result.params.iloc[0], result.params.iloc[1]
    return 1 / (1 + np.exp(-(b0 + b1 * np.asarray(net_profit, dtype=float))))
