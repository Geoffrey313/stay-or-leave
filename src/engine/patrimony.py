"""Valeur patrimoniale de cycle de vie Pi (assemblage de la verticale statique).

Pi réunit, à une position de distribution donnée et par pays, en valeur présente et en
dollars internationaux :
  - le FLUX de consommation réelle annuelle (pouvoir d'achat réel plafonné à la saturation),
    actualisé sur la carrière au taux rho ;
  - le STOCK terminal de retraite (bloc D5) : flux de cotisation composé à r_rep en répartition
    ou r_mkt en capitalisation ;
  - le STOCK terminal de surplus (bloc D6) : épargne au-dessus de la saturation, composée au
    rendement net de l'impôt sur les plus-values r_mkt*(1 - tau).
Les deux stocks terminaux sont ramenés en valeur présente au même taux rho (convention
déclarée, pré-Bellman), pour que flux et stocks soient homogènes.

RISQUE. Les deux jambes capitalisées (retraite capitalisée, surplus) portent le rendement
RISQUÉ r_mkt. On expose Pi_gross (au taux r_mkt, BORNE HAUTE avant prime de risque) et
Pi_ce (au taux équivalent-certain CRRA/Merton r_mkt - 0,5*gamma*sigma^2). L'ajustement de
risque canonique est fait UNE SEULE FOIS dans la Bellman ; ces deux versions l'encadrent.

CONVENTIONS déclarées :
  - Prime salariale NON intégrée (phi=1) : à position de destination égale, le migrant occupe
    le même percentile qu'à l'origine. C'est une hypothèse forte (un cadre P90 français ne
    devient pas mécaniquement un P90 américain) ; relâchée au bloc D7. Ce Pi n'est donc PAS
    encore la réponse « rester vs partir », c'est sa version à position égale.
  - Baseline LOCATAIRE : le loyer (bloc D4) est soustrait comme dépense pure (il réduit le
    surplus investissable) ; le canal PROPRIÉTÉ (épargne forcée, actif terminal) n'est pas
    encore intégré.
  - Imposition du capital à l'ACCRUAL (rendement taxé chaque année, r_mkt*(1 - tau)). Une
    imposition à la réalisation (plus-value terminale seule) différerait, surtout au sommet.
Grid-agnostique : opère sur les positions livrées.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from src.data.cascade import CascadePoint
from src.data.capital_tax import CapitalTax
from src.data.cost_of_living import CostOfLiving
from src.data.housing import Housing
from src.data.pension import Calibration, PensionParams
from src.engine.pension import fv_annuity_factor


def pv_annuity_factor(r: float, years: int) -> float:
    """Valeur présente d'un flux annuel constant unitaire sur `years`, au taux `r`."""
    return (1 - (1 + r) ** (-years)) / r


class Patrimony(BaseModel):
    """Valeur patrimoniale lifetime à une position, en dollars internationaux (valeur présente)."""
    country: str
    profile: str
    position: str
    real_pp_intl_usd: float    # pouvoir d'achat réel annuel (sert à la translation de rang, D7)
    consumption_pv: float
    pension_pv_gross: float
    surplus_pv_gross: float
    pi_gross: float            # borne haute avant prime de risque
    pi_ce: float               # équivalent-certain illustratif


def lifetime_patrimony(points: List[CascadePoint], col: Dict[str, CostOfLiving],
                       pension: Dict[str, PensionParams], tax: Dict[str, CapitalTax],
                       cal: Calibration, housing: Dict[str, Housing] | None = None) -> List[Patrimony]:
    """Assemble Pi par (profil, position) pour un pays, en valeur présente, $ internationaux.

    Baseline LOCATAIRE (D4) : si `housing` est fourni, le loyer réel annuel est soustrait du
    pouvoir d'achat réel avant le partage consommation/surplus (dépense pure, non capitalisée).
    """
    T, rho = cal.career_years, cal.discount_rate_rho
    sat = cal.consumption_saturation_intl_usd
    flow_factor = pv_annuity_factor(rho, T)
    to_pv = (1 + rho) ** (-T)
    out: List[Patrimony] = []
    for p in points:
        c = p.country
        ppp = col[c].ppp_aic_local_per_intl_usd
        real_pp = p.disposable_year / ppp                       # pouvoir d'achat réel ($ int.)
        rent_real = (housing[c].rent_monthly_local * 12 / ppp) if housing and c in housing else 0.0
        real_after_rent = max(0.0, real_pp - rent_real)         # loyer = dépense pure (locataire)
        consumption = min(real_after_rent, sat)
        surplus = max(0.0, real_after_rent - sat)
        consumption_pv = consumption * flow_factor

        # Stock retraite (D5), cotisation SCINDÉE : part répartition -> r_rep (sans risque) ;
        # part capitalisée -> r_mkt (risqué). Un pays à régime unique = une part nulle (inchangé) ;
        # un pays mixte (CH) somme les deux stocks à leurs rendements respectifs.
        contrib_payg = pension[c].payg_share * p.cost_employer / ppp      # en $ int.
        contrib_funded = pension[c].funded_share * p.cost_employer / ppp
        pension_pv_gross = (contrib_payg * fv_annuity_factor(cal.r_rep_real, T)
                            + contrib_funded * fv_annuity_factor(cal.r_mkt_real, T)) * to_pv
        pension_pv_ce = (contrib_payg * fv_annuity_factor(cal.r_rep_real, T)
                         + contrib_funded * fv_annuity_factor(cal.r_mkt_certainty_equivalent, T)) * to_pv

        # Stock surplus (D6) : investi à r_mkt net d'impôt sur les plus-values (accrual), risqué
        net = 1 - tax[c].capgains_tax_rate
        surplus_pv_gross = surplus * fv_annuity_factor(cal.r_mkt_real * net, T) * to_pv
        surplus_pv_ce = surplus * fv_annuity_factor(cal.r_mkt_certainty_equivalent * net, T) * to_pv

        out.append(Patrimony(
            country=c, profile=p.profile, position=p.position,
            real_pp_intl_usd=round(real_pp, 1),
            consumption_pv=round(consumption_pv),
            pension_pv_gross=round(pension_pv_gross),
            surplus_pv_gross=round(surplus_pv_gross),
            pi_gross=round(consumption_pv + pension_pv_gross + surplus_pv_gross),
            pi_ce=round(consumption_pv + pension_pv_ce + surplus_pv_ce),
        ))
    return out


class LifetimeComparison(BaseModel):
    """Rester vs partir en valeur patrimoniale lifetime (phi=1), à position égale."""
    profile: str
    position: str
    origin: str
    destination: str
    pi_origin_ce: float
    pi_destination_ce: float
    leave_gain_ce: float
    leave_dominates_ce: bool


def stay_or_leave_lifetime(origin: List[Patrimony],
                           destination: List[Patrimony]) -> List[LifetimeComparison]:
    """Compare Pi (équivalent-certain) origine vs destination sur les positions communes."""
    dest = {(p.profile, p.position): p for p in destination}
    out: List[LifetimeComparison] = []
    for o in origin:
        d = dest.get((o.profile, o.position))
        if d is None:
            continue
        gain = round(d.pi_ce - o.pi_ce)
        out.append(LifetimeComparison(
            profile=o.profile, position=o.position, origin=o.country, destination=d.country,
            pi_origin_ce=o.pi_ce, pi_destination_ce=d.pi_ce,
            leave_gain_ce=gain, leave_dominates_ce=gain > 0,
        ))
    return out
