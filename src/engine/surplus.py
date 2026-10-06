"""Surplus d'épargne libre capitalisé, net de fiscalité du capital (bloc D6).

Le revenu disponible réel au-dessus d'un seuil de saturation de la consommation est épargné
et investi. Il compose au rendement de marché RISQUÉ r_mkt, net de l'impôt sur les plus-values
du pays de résidence : rendement net = r_mkt * (1 - tau). Comme pour la retraite (D5), r_mkt
est une espérance d'actif risqué : le patrimoine terminal brut est une BORNE HAUTE AVANT
PRIME DE RISQUE ; Pi transporte r_mkt et sa variance, l'ajustement de risque canonique est
fait une fois dans la Bellman. L'équivalent-certain illustratif utilise le taux CRRA/Merton
(r_mkt - 0,5*gamma*sigma^2), net du même impôt.

Surplus annuel réel (dollars internationaux) = max(0, pouvoir d'achat réel - saturation).
Grid-agnostique : opère sur les positions livrées, apparie origine/destination sur les communes.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from src.data.capital_tax import CapitalTax
from src.data.pension import Calibration
from src.engine.pension import fv_annuity_factor
from src.engine.purchasing_power import RealPoint


class SurplusSwitch(BaseModel):
    """Surplus capitalisé et gain de le délocaliser, à une position (dollars internationaux)."""
    profile: str
    position: str
    origin: str
    destination: str
    surplus_origin: float
    surplus_destination: float
    wealth_gain_gross_pre_risk: float     # patrimoine terminal gagné, au taux r_mkt net d'impôt
    wealth_gain_ce_illustrative: float    # équivalent-certain approché (illustration de l'écart)


def _surplus(real_pp: float, saturation: float) -> float:
    return max(0.0, real_pp - saturation)


def surplus_switch(origin: List[RealPoint], destination: List[RealPoint],
                   tax: Dict[str, CapitalTax], cal: Calibration) -> List[SurplusSwitch]:
    """Patrimoine terminal du surplus investi, origine vs destination, par position commune."""
    dest = {(r.profile, r.position): r for r in destination}
    T = cal.career_years
    out: List[SurplusSwitch] = []
    for o in origin:
        d = dest.get((o.profile, o.position))
        if d is None:
            continue
        tau_o = tax[o.country].capgains_tax_rate
        tau_d = tax[d.country].capgains_tax_rate
        s_o = _surplus(o.real_pp_intl_usd, cal.consumption_saturation_intl_usd)
        s_d = _surplus(d.real_pp_intl_usd, cal.consumption_saturation_intl_usd)
        fv_gross = lambda tau: fv_annuity_factor(cal.r_mkt_real * (1 - tau), T)
        fv_ce = lambda tau: fv_annuity_factor(cal.r_mkt_certainty_equivalent * (1 - tau), T)
        gross = s_d * fv_gross(tau_d) - s_o * fv_gross(tau_o)
        ce = s_d * fv_ce(tau_d) - s_o * fv_ce(tau_o)
        out.append(SurplusSwitch(
            profile=o.profile, position=o.position, origin=o.country, destination=d.country,
            surplus_origin=round(s_o), surplus_destination=round(s_d),
            wealth_gain_gross_pre_risk=round(gross), wealth_gain_ce_illustrative=round(ce),
        ))
    return out
