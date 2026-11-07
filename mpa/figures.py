"""Every figure of the paper. Figures 3, 4, B1 from the supply step (earlier package, slopes_plots.py); Figures 5-8 and
B2 from Analysis/Econometric analysis.ipynb. Each function returns a matplotlib figure; save() writes .svg and .png."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import econometrics as em  # noqa: E402
from .supply import day_interval, day_to_date, fit_params, get_fit, predict, FIG4_REGIMES, MIN_SPACE_FIG_B, MIN_SPACE_PANEL  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig, out_dir: Path, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(Path(out_dir) / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- Figures 3, 4, B1: regimes and supply-curve fits ----------------------------------------------------------
def _style_quantity_axis(ax: Axes) -> None:
    ax.set_xlim(0, 80_000)
    ax.set_xticks([0, 20_000, 40_000, 60_000, 80_000])
    ax.set_xticklabels(["0", "20", "40", "60", "80"])


def _plot_regime_fit(ax: Axes, fits: pd.DataFrame, market: pd.DataFrame, regime: int, start: int, end: int, min_space: float) -> float:
    """Scatter + fit + breakpoints + R^2 for one regime. Returns R^2 weighted by the number of observations."""
    seg = day_interval(market, start, end)
    row = get_fit(fits, regime, min_space)
    ax.scatter(x=seg["quantity"], y=seg["price"], alpha=0.2, rasterized=True)
    x_space = np.linspace(seg["quantity"].min(), seg["quantity"].max(), num=100)
    ax.plot(x_space, predict(row, x_space), label="Fit", color="darkorange", lw=2)
    for bp in fit_params(row)[0]:
        ax.axvline(bp, ls=":", lw=1.5, color="black", alpha=0.8)
    ax.text(0.97, 0.03, f"$R^2 = {row['r_squared']:.3f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    return float(row["r_squared"]) * len(seg)


def fig3(fuel: pd.DataFrame, change_points: list[int]):
    """Fuel costs with the regimes (``project_regimes`` in notebook 05)."""
    inputs = [fuel["price_gas"].rename("Carbon-adjusted gas price"), fuel["price_coal"].rename("Carbon-adjusted coal price")]
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    max_value = 0.0
    colors = ["forestgreen", "dodgerblue"]
    for i, ser in enumerate(inputs):
        ax.plot(ser, label=ser.name, alpha=1, color=colors[i])
        max_value = max(max_value, ser.max())
    max_value += 10

    cps = day_to_date(change_points)
    for i in range(len(cps)):
        if i == len(cps) - 1:
            continue
        start = cps[i]
        ax.axvline(start, color="black", ls="--", alpha=1, lw=1)
        if i % 2 == 0:
            continue
        ax.fill_between([start, cps[i + 1]], 0, max_value, color="lightgrey", alpha=0.3)
    ax.set_xlim(pd.to_datetime("2019-01-01"), pd.to_datetime("2025-01-01"))
    ax.set_ylim(0, max_value)
    ax.grid(True, which="major", axis="y")
    ax.grid(False)
    ax.set_xlabel("Date")
    ax.set_ylabel("Carbon-adjusted fuel cost [€/MWh thermal]")
    ax.set_title("Identification of regime shifts in electricity supply")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1), frameon=False, shadow=False, ncol=1)
    fig.tight_layout()
    return fig


def fig4(market: pd.DataFrame, fits: pd.DataFrame, change_points: list[int], selected_clusters=FIG4_REGIMES, min_space: float = MIN_SPACE_PANEL):
    """Some slopes (``plot_some_slopes`` in notebook 05)."""
    fig, axss = plt.subplots(1, len(selected_clusters), figsize=(4 * len(selected_clusters), 7), sharex=True, sharey=True)
    for i, regime in enumerate(selected_clusters):
        start, end = change_points[regime - 1], change_points[regime]
        ax: Axes = axss[i]
        _style_quantity_axis(ax)
        _plot_regime_fit(ax, fits, market, regime, start, end, min_space)
        day_start, day_end = day_to_date([start, end])
        ax.set_title(f"Piecewise linear fit (Regime {regime})\n{day_start.date()} to {day_end.date()}")
        ax.grid(True)
    for ax in np.array(axss).flatten():
        ax.set_xlabel("Residual Load [GW]")
        ax.set_ylabel("Electricity Price [€/MWh]")
    fig.tight_layout()
    return fig


def figB1(market: pd.DataFrame, fits: pd.DataFrame, change_points: list[int], min_space: float = MIN_SPACE_FIG_B, ncols: int = 4):
    """Final slopes (``plot_slopes`` in notebook 05). Returns the figure and the average R^2."""
    n = len(change_points) - 1
    ncols = min(n, ncols)
    nrows = (1 if n % ncols else 0) + n // ncols
    fig, axss = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.6 * nrows))
    cumulative_r2 = 0.0
    for i in range(n):
        start, end = change_points[i], change_points[i + 1]
        ax: Axes = axss[i // ncols][i % ncols]
        _style_quantity_axis(ax)
        ax.set_ylim(market["price"].min() - 5, market["price"].max() + 5)
        cumulative_r2 += _plot_regime_fit(ax, fits, market, i + 1, start, end, min_space)
        ax.set_title(f"Regime {i + 1}, days {start + 1}-{end}")
        ax.grid(True)
    for i, ax in enumerate(np.array(axss).flatten()):
        if i > n - 1 - ncols:
            ax.set_xlabel("Quantity [GWh]")
        if i % ncols == 0:
            ax.set_ylabel("Price [€/MWh]")
    for ax in np.array(axss).flatten()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, cumulative_r2 / len(market)


# ---- Figures 5-8, B2: panel and logit results (Econometric analysis.ipynb) ----------------------------------
TECH_COLOR = {"Fossil Brown coal/Lignite": "steelblue", "Fossil Hard coal": "darkorange",
              "Fossil Gas gas/steam turbine": "forestgreen", "Fossil Gas Combined cycle": "springgreen"}


def fig5(df: pd.DataFrame, unit="11WD7GERS-G-BLIY", start="2024-04-01 00:00:00+00:00", end="2024-04-30 00:00:00+00:00"):
    """Company generation and hedged quantity of the company owning `unit`, April 2024."""
    d = df.xs(unit, level="EIC").loc[start:end]
    gen, hedged = d.company_gen.astype(float), d.hedged_quantity.astype(float)
    fig, ax = plt.subplots(figsize=(15, 8))
    gen.plot(ax=ax, label="Company generation", color="black", alpha=0.7)
    hedged.plot(ax=ax, label="Assumed hedged quantity", color="orangered", alpha=0.7, linewidth=2)
    ax.fill_between(d.index, gen, hedged, where=gen > hedged, facecolor="palegreen", alpha=0.5)
    ax.fill_between(d.index, gen, hedged, where=gen < hedged, facecolor="lightpink", alpha=0.5)
    ax.set_ylabel("Quantity [MWh per hour]", fontsize=14)
    ax.set_xlabel("Date, in hourly time step", fontsize=14)
    ax.set_title(f"Company generation and estimated net exposure, {d.company.iloc[0]}", fontsize=16)
    ax.legend(fontsize=14, loc="lower center", frameon=False)
    ax.grid(True, which="major", axis="y")
    ax.set_ylim(0, None)
    fig.tight_layout()
    return fig


def fig6(df: pd.DataFrame, unit="11WD2IRG4000061X", start="2024-03-01 00:00:00+00:00", end="2024-03-28 00:00:00+00:00"):
    """Net profit from withholding (top) and push-in (bottom) for one gas unit, March 2024; green where the
    benchmark allows the deviation."""
    d = df.xs(unit, level="EIC").loc[start:end]
    color = TECH_COLOR[d.production_type.iloc[0]]
    z = d[em.Z]
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    d[em.WITHHOLD].plot(ax=axs[0], color=color, alpha=0.5, linewidth=2, linestyle="--")
    d[em.PUSHIN].plot(ax=axs[1], label="Net profit, no opportunity", color=color, alpha=0.5, linewidth=2, linestyle="--")
    d[em.WITHHOLD].where(z == 1, 0).plot(ax=axs[0], color="green", alpha=0.7, linewidth=2)
    d[em.PUSHIN].where(z == 0, 0).plot(ax=axs[1], label="Net profit, with opportunity", color="green", alpha=0.7, linewidth=2)
    for ax in axs:
        ax.axhline(y=0, color="black", linewidth=1)
        ax.grid(True)
    axs[0].set_ylabel("Net profit from withholding [EUR/MW]", fontsize=12)
    axs[1].set_ylabel("Net profit from pushing in [EUR/MW]", fontsize=12)
    axs[1].set_xlabel("Date, in hourly time step", fontsize=14)
    axs[0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    axs[0].set_xlabel("")
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    t1 = pd.Timestamp("2024-03-09 17:00:00+00:00")
    axs[1].annotate("Incentive exists\nwhen net profit is positive", xy=(t1, d.loc[t1, em.PUSHIN]),
                    xytext=(t1 - pd.Timedelta(days=3), d.loc[t1, em.PUSHIN] + 58), fontsize=12,
                    arrowprops=dict(arrowstyle="->", lw=1), bbox=box)
    t2 = pd.Timestamp("2024-03-26 11:00:00+00:00")
    axs[0].annotate("High incentive\nindicated by high net profit", xy=(t2, d.loc[t2, em.WITHHOLD]),
                    xytext=(t2 - pd.Timedelta(days=8.5), d.loc[t2, em.WITHHOLD] - 4), fontsize=12,
                    arrowprops=dict(arrowstyle="->", lw=1), bbox=box)
    plt.legend(fontsize=12, loc="lower left", frameon=False)
    axs[0].set_title(f"Estimated net profit from deviating dispatch for a gas-fired unit of {d.company.iloc[0]}", fontsize=16)
    fig.tight_layout()
    return fig


def _fig7_predicted(df, model, var, size):
    """Predicted probability over an even grid of net profit (range of the whole panel), averaged in `size` groups."""
    grid = np.linspace(df[var].min(), df[var].max(), size)
    p = pd.DataFrame({var: grid, "p": em.predict(model, grid)})
    p["group"] = pd.qcut(p[var], q=size, labels=False, duplicates="drop")
    return p.groupby("group")[[var, "p"]].mean().sort_values(var)


def _fig7_observed(df, z, var, size):
    """Shares of observed deviation (-1, 0, 1) in `size` equal-width bins of net profit, model sample z."""
    d = df[df[em.Z] == z]
    bins = np.linspace(d[var].min(), d[var].max(), size + 1)
    g = pd.cut(d[var], bins=bins, labels=False, include_lowest=True).rename("group")
    y = d[em.Y].astype("category").cat.set_categories([0, -1, 1])
    counts = y.groupby(g, observed=False).value_counts().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares[var] = d[var].groupby(g).mean()
    return shares.sort_values(var)


def fig7(df: pd.DataFrame, main, size: int = 2000):
    """Predicted probability of deviation (lines, grey background) and observed shares (bars) along net profit."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, z, model, var, legend in ((axs[0], 1, main[0], em.WITHHOLD, False), (axs[1], 0, main[1], em.PUSHIN, True)):
        pred, obs = _fig7_predicted(df, model, var, size), _fig7_observed(df, z, var, size)
        x, p = pred[var], pred["p"]
        neg, pos = (p, 0 * p) if z == 1 else (0 * p, p)
        ax.fill_between(x, neg, 0, label="Negative deviation in generation (no data)" if legend else None, color="darkgrey")
        ax.fill_between(x, pos + neg, neg, label="Positive deviation in generation (no data)" if legend else None, color="dimgrey")
        ax.fill_between(x, 1, pos + neg, label="No significant deviation (no data)" if legend else None, color="gainsboro")
        if z == 1:
            ax.plot(x, p, label="Predicted probability of withholding", ls="-.", lw=1, color="k")
            ax.plot(x, [2] * len(x), label="Predicted probability of pushing in", ls=":", lw=2, color="k")
            ax.plot(x, [2] * len(x), label="   ", ls="-", lw=1, color="white")
        else:
            ax.plot(x, p, label="", ls=":", lw=2, color="k")
        width = (obs[var].max() - obs[var].min()) / size
        ax.bar(obs[var], obs[1], bottom=obs[-1], label="Positive deviation in generation" if legend else None, color="dodgerblue", width=width)
        ax.bar(obs[var], obs[-1], label="Negative deviation in generation" if legend else None, color="darkorange", width=width)
        ax.bar(obs[var], obs[0], bottom=obs[-1] + obs[1], label="No significant deviation" if legend else None, color="lightgreen", width=width)
        ax.set_xlabel(f"Net profit from capacity {'withholding' if z == 1 else 'push-in'} [EUR/MW]")
        ax.set_ylabel("Relative instances of deviation")
        ax.set_title("Data and model fit given baseline dispatch" if z == 1 else "Data and model fit given baseline non-dispatch")
        ax.set_ylim(0, 1)
    axs[0].set_xlim(-850, 250)
    axs[1].set_xlim(-950, 250)
    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def coefficients(models: dict) -> pd.DataFrame:
    """Coefficient of net profit, 90 % confidence half-width and p-value per group (data behind Figures 8 and B2)."""
    rec = []
    for g, pair in models.items():
        for name, m in zip(("withholding", "push-in"), pair):
            lo, hi = m.conf_int(alpha=0.1).iloc[1]
            rec.append((g, name, m.params.iloc[1], hi - m.params.iloc[1], m.pvalues.iloc[1]))
    return pd.DataFrame(rec, columns=["group", "model", "coef", "ci90", "p"])


def _bars(ax, c: pd.DataFrame, labels, width):
    x = np.arange(len(labels))
    for k, (name, color, shift) in enumerate((("withholding", "darkorange", -1), ("push-in", "dodgerblue", 1))):
        s = c[c.model == name]
        ax.bar(x + shift * width / 2, s.coef, width=width, yerr=s.ci90,
               color=[color if p <= 0.1 else "#A9A9A9" for p in s.p],
               label=("Effect of profitable withholding on negative deviation" if k == 0 else
                      "Effect of profitable push-in on positive deviation"))
    ax.set_xticks(x)
    ax.axhline(y=0, color="darkgray", linestyle="-", alpha=0.5)
    return x


def fig8(coef: pd.DataFrame):
    """Company models, companies ordered by size (largest absolute net exposure)."""
    companies = list(dict.fromkeys(coef.group))
    fig, ax = plt.subplots(figsize=(12, 6))
    _bars(ax, coef, companies, 0.35)
    ax.set_xticklabels(companies, rotation=45, ha="right")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_ylabel("Parameter value", fontsize=14)
    ax.set_xlabel("Generation company", fontsize=14)
    ax.set_title("Estimated effect of net profit of per MW withholding and push-in on deviation in dispatch", fontsize=14)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [Patch(facecolor="#A9A9A9", label="Insignificant estimate (p-value > 0.1)")],
              loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def figB2(coef: pd.DataFrame):
    """Technology models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _bars(ax, coef, list(em.TECHS), 0.25)
    ax.set_xticklabels(list(em.TECHS.values()), fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title("Estimated effect of net profit of market power abuse on deviation in dispatch,\n by production type", fontsize=14)
    patches = [Patch(facecolor="darkorange", label="Effect of net profit from capacity withholding on observed negative deviation in dispatch"),
               Patch(facecolor="dodgerblue", label="Effect of net profit from capacity push-in on observed positive deviation in dispatch"),
               Patch(facecolor="#A9A9A9", label="Insignificant estimate (p-value > 0.1)")]
    ax.legend(handles=patches, bbox_to_anchor=(0.9, -0.1), frameon=False, fontsize=12)
    ax.grid(linewidth=0.5)
    ax.set_ylim(-0.015, 0.085)
    fig.tight_layout()
    return fig
