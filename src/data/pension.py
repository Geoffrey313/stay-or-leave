"""Lecture des paramètres retraite (bloc D5) et de la calibration structurelle.

Entrées :
  - data/params/pension.csv : régime (payg|funded) et part de cotisation retraite par pays.
  - data/params/calibration.csv : r_rep, r_mkt, horizon de carrière (hypothèses déclarées).
"""
from __future__ import annotations

import csv
from typing import Dict

from pydantic import BaseModel, NonNegativeFloat

from src.common.paths import param_file


class PensionParams(BaseModel):
    """Régime et part de cotisation retraite d'un pays (part du coût employeur)."""
    country: str
    regime: str  # payg | funded
    contrib_share_cost: NonNegativeFloat


class Calibration(BaseModel):
    """Calibration structurelle (hypothèses déclarées)."""
    r_rep_real: float
    r_mkt_real: float
    career_years: int


def _read(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(filter(lambda l: not l.startswith("#"), fh)))


def load_pension() -> Dict[str, PensionParams]:
    rows = _read(param_file("pension.csv"))
    return {r["country"]: PensionParams(country=r["country"], regime=r["regime"],
                                        contrib_share_cost=float(r["contrib_share_cost"]))
            for r in rows}


def load_calibration() -> Calibration:
    d = {r["param"]: r["value"] for r in _read(param_file("calibration.csv"))}
    return Calibration(r_rep_real=float(d["r_rep_real"]), r_mkt_real=float(d["r_mkt_real"]),
                       career_years=int(float(d["career_years"])))
