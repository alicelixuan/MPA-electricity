"""Step 6: every table of the paper from the analysis panel. Each function returns a tidy frame
(row, column, value, stars); display() pivots it
into the layout of the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import econometrics as em

# Scenario inputs of Tables 3 and B1: (row, slope [EUR/MWh per MW], net exposure [MW], margin [EUR/MWh])
SCENARIOS_WITHHOLD = [("i", 0.008, 800, 100), ("ii", 0.008, 800, 1), ("iii", 0.04, 800, 1), ("iv", 0.04, 4525, 1)]
SCENARIOS_PUSHIN = [("i", 0.006, -800, -100), ("ii", 0.006, -800, -10), ("iii", 0.02, -800, -5), ("iv", 0.02, -9000, -5)]
REDISPATCH = ("redispatch_decrease_MWh", "redispatch_increase_MWh")


def stars(p: float) -> str:
    """Significance as printed in the paper: the 10 % level is applied to p rounded to one decimal."""
    return "***" if p <= 0.01 else "**" if p <= 0.05 else "*" if round(p, 1) <= 0.1 else ""


def _rows(records) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["row", "column", "value", "stars"])


def _logit_rows(models, labels) -> pd.DataFrame:
    """Rows of a regression table: one column per model, coefficients with stars, McFadden R2 (%), N."""
    rec = []
    for col, m in zip(("withholding", "push-in"), models):
        for name, b in m.params.items():
            rec.append((labels.get(name, name), col, b, stars(m.pvalues[name])))
        rec += [("R2 McFadden [%]", col, 100 * m.prsquared, ""), ("Observations", col, m.nobs, "")]
    return _rows(rec)


LABELS = {"const": "Intercept", em.WITHHOLD: "Net profit from withholding [EUR/MW]",
          em.PUSHIN: "Net profit from push-in [EUR/MW]",
          "net_profit_withhold_third_unit": "Net profit of withholding 1/3 unit [EUR]",
          "net_profit_pushin_third_unit": "Net profit of pushing in 1/3 unit [EUR]",
          REDISPATCH[0]: "Downward redispatch volume [MWh]", REDISPATCH[1]: "Upward redispatch volume [MWh]"}


def table1(df):
    """Quantiles of net profit, each over its model sample (withholding: z = 1, push-in: z = 0)."""
    rec = []
    for name, d, col in (("From withholding [EUR/MW]", df[df[em.Z] == 1], em.WITHHOLD),
                         ("From push-in [EUR/MW]", df[df[em.Z] == 0], em.PUSHIN)):
        for q, lab in ((0, "Minimum"), (0.5, "50 %"), (0.9, "90 %"), (0.99, "99 %"), (1, "Maximum")):
            rec.append((name, lab, d[col].quantile(q), ""))
        rec.append((name, "Share profitable [%]", 100 * (d[col] > 0).mean(), ""))   # text, Section 5.3
    return _rows(rec)


def table2(main):
    return _logit_rows(main, LABELS)


def _scenarios(df, model, scenarios, sign, col, sample):
    """Tables 3 and B1: net profit of each scenario (Eqs. 9-10), its occurrence, and the probability predicted by the
    main model. Occurrence = share of the model sample with at least the scenario's net profit as printed (whole EUR).
    Row iv is built from rounded parameters to match the largest observation (e.g. 174.9 -> 175), so the threshold is
    capped at the sample maximum."""
    rec = []
    for row, slope, exposure, margin in scenarios:
        npf = sign * (slope * exposure - margin)
        threshold = min(round(npf), sample[col].max())
        rec += [(row, "Slope", slope, ""), (row, "Net exposure [MW]", exposure, ""), (row, "Margin [EUR/MW]", margin, ""),
                (row, "Net profit [EUR/MW]", npf, ""),
                (row, "Occurrence [%]", 100 * (sample[col] >= threshold).mean(), ""),
                (row, "Predicted probability [%]", 100 * em.predict(model, npf).item(), "")]
    return _rows(rec)


def table3(df, main):
    return _scenarios(df, main[0], SCENARIOS_WITHHOLD, 1, em.WITHHOLD, df[df[em.Z] == 1])


def tableB1(df, main):
    return _scenarios(df, main[1], SCENARIOS_PUSHIN, -1, em.PUSHIN, df[df[em.Z] == 0])


def _coef_rows(models_by_col):
    rec = []
    for col, (m1, m2) in models_by_col.items():
        rec += [("Net profit from withholding [EUR/MW]", col, m1.params.iloc[1], stars(m1.pvalues.iloc[1])),
                ("Net profit from push-in [EUR/MW]", col, m2.params.iloc[1], stars(m2.pvalues.iloc[1]))]
    return _rows(rec)


def table4(df):
    years = df.index.get_level_values("DateTime").year
    return _coef_rows(em.by_group(df, lambda d: d.index.get_level_values("DateTime").year, sorted(set(years))))


def table5(df):
    return _logit_rows(em.regime_switching_logit(df, (em.WITHHOLD, *REDISPATCH), (em.PUSHIN, *REDISPATCH)), LABELS)


def table6(df, main):
    out = {"100 %": main}
    for r in em.HEDGE_RATES[1:]:
        out[f"{100 * r:g} %"] = em.regime_switching_logit(df, (f"{em.WITHHOLD}_hedge_{r:g}",), (f"{em.PUSHIN}_hedge_{r:g}",))
    t = _coef_rows(out)
    return t.assign(row=t.column, column=t.row)        # hedge rates as rows, as in the paper


def table7(df):
    return _logit_rows(em.regime_switching_logit(df, ("net_profit_withhold_third_unit",), ("net_profit_pushin_third_unit",)),
                       LABELS)


def tableB2(df, main, load: pd.Series):
    """Expected volume withheld / pushed in (predicted probability x available capacity, summed per hour) as a share of
    load in that hour, averaged over hours, and the implied hourly price change (volume x slope), averaged over hours.
    Notebook 8, cells 52-61."""
    d = df.reset_index()
    rec = []
    for key, z, model, col, sign in (("withheld", 1, main[0], em.WITHHOLD, 1), ("pushed in", 0, main[1], em.PUSHIN, -1)):
        x = d[d[em.Z] == z].assign(mw=lambda q: em.predict(model, q[col]) * q.available_capacity)
        h = x.groupby("DateTime").agg(mw=("mw", "sum"), slope=("slope", "mean"))
        h["dp"] = sign * h.mw * h.slope
        h["load"] = load.reindex(h.index).to_numpy()
        ext = "max" if sign == 1 else "min"
        for period, g in [("Six-year sample", h)] + list(h.groupby(h.index.year)):
            rec += [(str(period), f"Volume {key} / load [%]", 100 * (g.mw / g.load).mean(), ""),
                    (str(period), f"Average price change, {key} [EUR/MWh]", g.dp.mean(), ""),
                    (str(period), f"{ext.capitalize()} price change, {key} [EUR/MWh]", g.dp.agg(ext), "")]
    return _rows(rec)


def tableB3(df):
    """Share of unit-hours by deviation; unit-hours without a discretized benchmark count as no deviation.
    As printed, each row sums to 100: negative deviation = 100 - the other two shares rounded to 0.1."""
    rec = []
    for name, g in [("All units pooled", df)] + list(df.groupby("production_type")):
        y = g[em.Y]
        name = em.TECHS.get(name, name)
        none, pos = round(100 * ((y == 0) | y.isna()).mean(), 1), round(100 * (y == 1).mean(), 1)
        rec += [(name, "Negative deviation [%]", round(100 - none - pos, 1), ""),
                (name, "No significant deviation [%]", none, ""),
                (name, "Positive deviation [%]", pos, "")]
    return _rows(rec)


def all_tables(df: pd.DataFrame, load: pd.Series) -> dict[str, pd.DataFrame]:
    df = em.add_variants(df)
    main = em.regime_switching_logit(df)
    return {"Table_1": table1(df), "Table_2": table2(main), "Table_3": table3(df, main), "Table_4": table4(df),
            "Table_5": table5(df), "Table_6": table6(df, main), "Table_7": table7(df),
            "Table_B1": tableB1(df, main), "Table_B2": tableB2(df, main, load), "Table_B3": tableB3(df)}


def display(t: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Paper layout: rows x columns, numbers rounded, stars appended."""
    def fmt(v, s):
        if isinstance(v, (int, np.integer)) or float(v).is_integer() and abs(v) >= 100:
            return f"{v:,.0f}{s}"
        return (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.{digits}g}") + s
    t = t.assign(text=[fmt(v, s) for v, s in zip(t.value, t.stars)])
    last = ["R2 McFadden [%]", "Observations"]
    rows = [r for r in t.row.unique() if r not in last] + [r for r in last if r in set(t.row)]
    out = t.pivot(index="row", columns="column", values="text").reindex(index=rows, columns=t.column.unique())
    return out.fillna("").rename_axis(index=None, columns=None)


MIN_SHARE = {"Fossil Brown coal/Lignite": 0.40, "Fossil Hard coal": 0.40, "Fossil Gas Combined cycle": 0.45,
             "Fossil Gas gas/steam turbine": 0.20}       # minimum dispatch, share of capacity (dispatch model)


def tableA(df: pd.DataFrame, dispatch: pd.DataFrame, unit_params: pd.DataFrame) -> pd.DataFrame:
    """Appendix A: mean, median, min, max of the dispatch model's inputs and decision variables over all unit-hours of
    the 40 units (before the benchmark is discretized). Unit parameters (installed capacity K_i, minimum dispatch) are
    the dispatch model's, one value per unit. State, dispatch and startup are averages over the Monte Carlo iterations."""
    d = df.reset_index().merge(dispatch, on=["EIC", "DateTime"], how="left", validate="1:1")
    units = unit_params.set_index("EIC").loc[d.EIC.unique()]
    series = {"Price [EUR/MWh]": d.price, "Variable cost [EUR/MWh]": d.price - d.margin,
              "Startup cost [1000 EUR]": d.startup_cost / 1000, "Installed capacity [MW]": units.installed_capacity_model,
              "Minimum dispatch [MW]": units.production_type.map(MIN_SHARE) * units.installed_capacity_model,
              "Dispatch [MW]": d.dispatch.clip(lower=0), "State [-]": d.state, "Startup [-]": d.startup}
    rec = []
    for name, s in series.items():
        rec += [(name, "Mean", s.mean(), ""), (name, "Median", s.median(), ""), (name, "Min", s.min(), ""), (name, "Max", s.max(), "")]
    return _rows(rec)
