"""Benchmark statique « rester vs partir » (première verticale du modèle).

À position de distribution et profil comparables, confronte le pouvoir d'achat réel d'un
agent qui reste dans son pays d'origine à celui du même agent au pays de destination. C'est
la forme statique, annuelle et sans coût de migration, du cœur du modèle : un premier test
que la chaîne θ -> coût de la vie -> pouvoir d'achat réel -> comparaison s'assemble.

Grid-agnostique : apparie origine et destination sur les positions ET profils présents dans
les DEUX pays. Les pays à résolution grossière (points OCDE) traversent sans adaptation ;
l'appariement se fait sur l'intersection des positions. Cette version compare à position
égale (même percentile dans chaque pays) ; la prime salariale (quelle position on occupe à
l'étranger) et le coût de migration sont des raffinements des blocs suivants.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.engine.purchasing_power import RealPoint


class Comparison(BaseModel):
    """Écart de pouvoir d'achat réel rester (origine) vs partir (destination), à position égale."""
    profile: str
    position: str
    origin: str
    destination: str
    real_pp_origin: float
    real_pp_destination: float
    leave_gain_intl_usd: float       # destination - origine
    leave_dominates: bool


def stay_or_leave(origin: List[RealPoint], destination: List[RealPoint]) -> List[Comparison]:
    """Compare origine et destination sur les (profil, position) communs."""
    dest = {(r.profile, r.position): r for r in destination}
    out: List[Comparison] = []
    for o in origin:
        d = dest.get((o.profile, o.position))
        if d is None:
            continue
        gain = round(d.real_pp_intl_usd - o.real_pp_intl_usd, 1)
        out.append(Comparison(
            profile=o.profile, position=o.position,
            origin=o.country, destination=d.country,
            real_pp_origin=o.real_pp_intl_usd, real_pp_destination=d.real_pp_intl_usd,
            leave_gain_intl_usd=gain, leave_dominates=gain > 0,
        ))
    return out
