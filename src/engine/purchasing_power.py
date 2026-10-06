"""Pouvoir d'achat réel (bloc D3 appliqué) : revenu disponible déflaté par le coût de la vie.

Déflate le revenu disponible nominal local par la parité de pouvoir d'achat sur la
consommation (TTC), pour obtenir un pouvoir d'achat réel comparable entre pays, en dollars
internationaux. Grid-agnostique : itère sur les positions livrées, quel qu'en soit le nombre.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from src.data.cascade import CascadePoint
from src.data.cost_of_living import CostOfLiving


class RealPoint(BaseModel):
    """Pouvoir d'achat réel à une position, en dollars internationaux."""
    country: str
    profile: str
    position: str
    cost_employer: int
    disposable_year: int
    real_pp_intl_usd: float          # revenu disponible / PPA (consommation, TTC)


def real_purchasing_power(points: List[CascadePoint],
                          col: Dict[str, CostOfLiving]) -> List[RealPoint]:
    """Déflate le revenu disponible de chaque point par la PPA de son pays."""
    out: List[RealPoint] = []
    for p in points:
        ppp = col[p.country].ppp_aic_local_per_intl_usd
        out.append(RealPoint(
            country=p.country, profile=p.profile, position=p.position,
            cost_employer=p.cost_employer, disposable_year=p.disposable_year,
            real_pp_intl_usd=round(p.disposable_year / ppp, 1),
        ))
    return out
