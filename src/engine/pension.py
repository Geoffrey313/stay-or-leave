"""Bascule retraite charge -> actif (bloc D5), le moteur économique du modèle.

En répartition, le flux retraite capté rend un rendement implicite faible r_rep (proche de
la croissance de la masse salariale). En capitalisation, le même flux devient un actif qui
compose au rendement de marché r_mkt. La valeur de la bascule, pour un agent qui quitte un
pays de répartition pour un pays de capitalisation, est son flux de cotisation retraite
revalorisé de r_rep à r_mkt, composé sur l'horizon de carrière restant.

Patrimoine terminal d'un flux annuel constant F sur T années au taux r :
    W(r, T) = F * ((1+r)^T - 1) / r.
Gain de bascule (origine répartition -> destination capitalisation) :
    F * [ W-facteur(r_mkt, T) - W-facteur(r_rep, T) ],
avec F = part de cotisation retraite x coût employeur à la position considérée.
Grid-agnostique : opère sur les positions livrées. Si l'origine n'est pas en répartition ou
la destination pas en capitalisation, la bascule est nulle.
"""
from __future__ import annotations

from typing import Dict, List

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
    annual_contribution: float       # flux retraite capté à cette position
    switch_gain_lifetime: float      # patrimoine terminal gagné par la bascule


def pension_switch(origin_points: List[CascadePoint], origin: PensionParams,
                   destination: PensionParams, cal: Calibration) -> List[PensionSwitch]:
    """Gain de bascule par position pour un agent quittant `origin` vers `destination`."""
    funded = origin.regime == "payg" and destination.regime == "funded"
    spread = (fv_annuity_factor(cal.r_mkt_real, cal.career_years)
              - fv_annuity_factor(cal.r_rep_real, cal.career_years))
    out: List[PensionSwitch] = []
    for p in origin_points:
        contrib = origin.contrib_share_cost * p.cost_employer
        gain = contrib * spread if funded else 0.0
        out.append(PensionSwitch(
            profile=p.profile, position=p.position,
            origin=origin.country, destination=destination.country,
            annual_contribution=round(contrib),
            switch_gain_lifetime=round(gain),
        ))
    return out
