"""Bascule retraite charge -> actif (bloc D5), le moteur économique du modèle.

En répartition, le flux retraite capté rend un rendement implicite faible r_rep, quasi SANS
RISQUE (proche de la croissance de la masse salariale). En capitalisation, le même flux
devient un actif RISQUÉ dont l'espérance de rendement est r_mkt > r_rep, avec volatilité
sigma. La valeur de la bascule, pour un agent qui quitte un pays de répartition pour un pays
de capitalisation, est son flux de cotisation retraite revalorisé de r_rep à r_mkt, composé
sur l'horizon de carrière.

RISQUE. r_mkt est une ESPÉRANCE d'actif risqué, pas un rendement certain. Le gain brut
ci-dessous (au taux r_mkt) est une BORNE HAUTE AVANT PRIME DE RISQUE, pas « l'avantage » de
la capitalisation. L'ajustement de risque canonique est fait UNE SEULE FOIS, dans la Bellman,
qui reçoit le rendement espéré r_mkt ET sa variance sigma^2. On expose ici, à titre
d'illustration de l'écart, l'équivalent-certain approché au taux CRRA/Merton
r_mkt - 0,5*gamma*sigma^2 (patrimoine terminal lognormal) ; ce n'est pas ce qui alimente Pi.

Patrimoine terminal d'un flux annuel constant F sur T années au taux r :
    W(r, T) = F * ((1+r)^T - 1) / r.

Deux réserves déclarées :
  - r_mkt plat = moyenne de rendement pondérée par le glide path 85->30 ; le stock terminal
    ignore l'effet de séquence (acceptable en premier passage, nommé comme tel).
  - la bascule fige la cotisation d'ORIGINE (origin.contrib_share * coût) et ne swappe que le
    rendement : elle ISOLE le canal rendement, qui est la thèse. Le changement de taux de
    cotisation entre pays (p.ex. FR 19,6 % -> US 12,4 %) est une marge distincte, non traitée ici.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.data.cascade import CascadePoint
from src.data.pension import Calibration, PensionParams


def fv_annuity_factor(r: float, years: int) -> float:
    """Facteur de valeur future d'un flux annuel constant unitaire."""
    return ((1 + r) ** years - 1) / r


class PensionSwitch(BaseModel):
    """Valeur de la bascule retraite à une position, en monnaie locale de l'origine."""
    profile: str
    position: str
    origin: str
    destination: str
    annual_contribution: float               # flux retraite capté (cotisation d'origine)
    switch_gain_gross_pre_risk: float        # au taux r_mkt : BORNE HAUTE avant risque (entre dans Pi avec sigma)
    switch_gain_ce_illustrative: float       # équivalent-certain approché (illustration de l'écart)


def pension_switch(origin_points: List[CascadePoint], origin: PensionParams,
                   destination: PensionParams, cal: Calibration) -> List[PensionSwitch]:
    """Gain de bascule par position pour un agent quittant `origin` vers `destination`.

    Le brut (r_mkt) est ce qui remonte dans Pi, accompagné de sigma pour que la Bellman fasse
    l'ajustement de risque une fois ; l'EC illustratif montre l'ordre de grandeur de l'écart.
    """
    is_switch = origin.regime == "payg" and destination.regime == "funded"
    w_rep = fv_annuity_factor(cal.r_rep_real, cal.career_years)              # sans risque
    w_mkt = fv_annuity_factor(cal.r_mkt_real, cal.career_years)              # espérance risquée
    w_ce = fv_annuity_factor(cal.r_mkt_certainty_equivalent, cal.career_years)  # équivalent-certain
    out: List[PensionSwitch] = []
    for p in origin_points:
        payg_loc, funded_loc = origin.split(p.cost_employer, 1.0)  # capé per-position, monnaie locale
        contrib = payg_loc + funded_loc
        gross = contrib * (w_mkt - w_rep) if is_switch else 0.0
        ce = contrib * (w_ce - w_rep) if is_switch else 0.0
        out.append(PensionSwitch(
            profile=p.profile, position=p.position,
            origin=origin.country, destination=destination.country,
            annual_contribution=round(contrib),
            switch_gain_gross_pre_risk=round(gross),
            switch_gain_ce_illustrative=round(ce),
        ))
    return out
