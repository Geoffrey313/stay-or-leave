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
    """Cotisation retraite d'un pays, SCINDÉE par régime (part du coût employeur) :
    part RÉPARTITION (-> droit W_p, rendement r_rep) et part CAPITALISÉE (-> pot W_f, r_mkt).
    Un pays à régime unique est le cas particulier où une des deux parts est nulle (FR répartition
    pure, US/UK capitalisation pure) ; un pays mixte porte les deux (CH : AVS répartition + BVG
    capitalisé). Le routage du split est fait par la Bellman à deux accumulateurs."""
    country: str
    payg_share: NonNegativeFloat     # cotisation répartition / coût employeur
    funded_share: NonNegativeFloat   # cotisation capitalisée / coût employeur

    @property
    def contrib_share_cost(self) -> float:
        """Cotisation retraite totale (répartition + capitalisée), pour compat statique."""
        return self.payg_share + self.funded_share

    @property
    def regime(self) -> str:
        """Régime DOMINANT (compat avec le switch statique binaire D5) : la vraie scission vit
        dans payg_share/funded_share et n'est utilisée que par le dynamique."""
        return "funded" if self.funded_share > self.payg_share else "payg"


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
    retirement_years: int

    @property
    def r_mkt_certainty_equivalent(self) -> float:
        """Taux équivalent-certain approché (CRRA/Merton) : r_mkt - 0,5*gamma*sigma^2."""
        return self.r_mkt_real - 0.5 * self.risk_aversion * self.sigma_mkt_real ** 2


def _read(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(filter(lambda l: not l.startswith("#"), fh)))


def load_pension() -> Dict[str, PensionParams]:
    rows = _read(param_file("pension.csv"))
    return {r["country"]: PensionParams(country=r["country"],
                                        payg_share=float(r["payg_share"]),
                                        funded_share=float(r["funded_share"]))
            for r in rows}


def load_calibration() -> Calibration:
    d = {r["param"]: r["value"] for r in _read(param_file("calibration.csv"))}
    return Calibration(r_rep_real=float(d["r_rep_real"]), r_mkt_real=float(d["r_mkt_real"]),
                       sigma_mkt_real=float(d["sigma_mkt_real"]),
                       risk_aversion=float(d["risk_aversion"]),
                       career_years=int(float(d["career_years"])),
                       consumption_saturation_intl_usd=float(d["consumption_saturation_intl_usd"]),
                       subsistence_floor_intl_usd=float(d["subsistence_floor_intl_usd"]),
                       discount_rate_rho=float(d["discount_rate_rho"]),
                       retirement_years=int(float(d["retirement_years"])))
