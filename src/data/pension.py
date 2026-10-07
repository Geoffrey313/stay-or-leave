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
    """Règle retraite CAPÉE d'un pays (bloc D5), depuis des paramètres statutaires sourcés. La
    cotisation n'est PAS une part plate du coût (fausse au sommet de tout système plafonné) : elle
    se calcule par position via `split`. Pilier RÉPARTITION : taux sur le brut, assiette plafonnée à
    payg_base_cap, DROIT (W_p, r_rep) plafonné à payg_benefit_cap (au-delà = redistribution,
    rendement ~0). Pilier CAPITALISÉ : taux sur la tranche [funded_floor, funded_cap] du brut
    (-> W_f, r_mkt) ; au-delà = surobligatoire volontaire = surplus (D6), jamais re-compté."""
    country: str
    employer_ssc_rate: NonNegativeFloat   # charges patronales / brut : brut = cost/(1+taux)
    payg_rate: NonNegativeFloat
    payg_base_cap: NonNegativeFloat
    payg_benefit_cap: NonNegativeFloat
    funded_rate: NonNegativeFloat
    funded_floor: NonNegativeFloat
    funded_cap: NonNegativeFloat

    def split(self, cost_employer: float, ppp: float) -> tuple[float, float]:
        """(cotisation vers W_p répartition, vers W_f capitalisé), en $ int, pour une position de
        coût employeur donné. Le brut est dérivé du coût ; les plafonds mordent au sommet."""
        gross = cost_employer / (1.0 + self.employer_ssc_rate)
        payg_to_wp = self.payg_rate * min(gross, self.payg_benefit_cap) / ppp
        coord = min(max(gross - self.funded_floor, 0.0),
                    max(self.funded_cap - self.funded_floor, 0.0))
        funded_to_wf = self.funded_rate * coord / ppp
        return payg_to_wp, funded_to_wf

    @property
    def regime(self) -> str:
        """Régime DOMINANT par taux (compat du switch statique binaire D5) ; la vraie règle capée
        vit dans `split` et n'est utilisée que par le dynamique et le patrimoine statique."""
        return "funded" if self.funded_rate > self.payg_rate else "payg"


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
    rows = _read(param_file("pension_rules.csv"))
    return {r["country"]: PensionParams(
        country=r["country"], employer_ssc_rate=float(r["employer_ssc_rate"]),
        payg_rate=float(r["payg_rate"]), payg_base_cap=float(r["payg_base_cap"]),
        payg_benefit_cap=float(r["payg_benefit_cap"]), funded_rate=float(r["funded_rate"]),
        funded_floor=float(r["funded_floor"]), funded_cap=float(r["funded_cap"]))
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
