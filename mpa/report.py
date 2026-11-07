"""Notebook display of the outputs in output/paper (and of a rerun comparison)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

from . import config, tables

ITEMS = {  # paper order
    "Figure_3": "Carbon-adjusted gas and coal prices and the 11 supply regimes",
    "Figure_4": "Piecewise linear supply-curve fits, regimes 1 and 8",
    "Figure_5": "Company generation and hedged quantity (net exposure), RWE, April 2024",
    "Table_1": "Net profit from capacity withholding and push-in (and the share of profitable unit-hours, Section 5.3)",
    "Figure_6": "Hourly net profit from withholding and push-in, Uniper gas unit, March 2024",
    "Table_2": "Main logit model",
    "Figure_7": "Predicted probabilities and observed deviation along net profit",
    "Table_3": "Predicted probability of withholding in four scenarios",
    "Table_4": "Logit models by year",
    "Figure_8": "Company-specific logit models",
    "Table_5": "Logit model with redispatch volume",
    "Table_6": "Hedge rate sensitivity",
    "Table_7": "Net profit of 1/3-unit deviation",
    "Table_A": "Dispatch model inputs and decision variables (Appendix A)",
    "Figure_B1": "Piecewise linear fits, all 11 regimes",
    "Table_B1": "Predicted probability of push-in in four scenarios",
    "Table_B2": "Expected volume withheld and pushed in, and price effects",
    "Table_B3": "Deviation in dispatch by production type",
    "Figure_B2": "Technology-specific logit models",
}


def show(item: str, out: Path = config.PAPER) -> None:
    display(Markdown(f"**{item.replace('_', ' ')}.** {ITEMS[item]}"))
    if item.startswith("Figure"):
        display(Image(filename=str(out / f"{item}.png"), width=900))
        return
    t = pd.read_csv(out / f"{item}.csv", keep_default_na=False)
    t["value"] = t.value.astype(float)
    display(tables.display(t))


def show_rerun(result: dict) -> None:
    if "rerun_dir" in result:
        display(Markdown(Path(result["rerun_dir"], "comparison.md").read_text()))
