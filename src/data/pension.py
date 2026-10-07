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
    """Calibration structurelle (hypothèses déclarées).

    r_mkt_real est l'ESPÉRANCE d'un rendement risqué (moyenne pondérée par le glide path) ;
    sigma_mkt_real sa volatilité et risk_aversion l'aversion CRRA, pour l'ajustement de risque.
    """
    r_rep_real: float
    r_mkt_real: float
    sigma_mkt_real: float
    risk_aversion: float
    career_years: int
    consumption_saturation_intl_usd: float
    subsistence_floor_intl_usd: float
    discount_rate_rho: float

    @property
    def r_mkt_certainty_equivalent(self) -> float:
        """Taux équivalent-certain approché (CRRA/Merton) : r_mkt - 0,5*gamma*sigma^2."""
        return self.r_mkt_real - 0.5 * self.risk_aversion * self.sigma_mkt_real ** 2


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
                       sigma_mkt_real=float(d["sigma_mkt_real"]),
                       risk_aversion=float(d["risk_aversion"]),
                       career_years=int(float(d["career_years"])),
                       consumption_saturation_intl_usd=float(d["consumption_saturation_intl_usd"]),
                       subsistence_floor_intl_usd=float(d["subsistence_floor_intl_usd"]),
                       discount_rate_rho=float(d["discount_rate_rho"]))
