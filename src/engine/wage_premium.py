"""Prime salariale phi (bloc D7) : translation de rang, nette du niveau déjà ancré.

L'ancrage de chaque cascade sur SA distribution porte DÉJÀ l'écart de niveau salarial entre
pays. phi ne réapplique donc pas un facteur de niveau (ce serait un double-compte) : il
translate le RANG. Un agent à une position d'origine, de pouvoir d'achat réel y, gagne à
l'étranger une rémunération à rôle comparable y * role_ratio_ppp (en réel), et atterrit au
percentile de DESTINATION qui correspond à ce revenu réel. On évalue alors la valeur
patrimoniale Pi de destination à ce percentile d'atterrissage (interpolée).

Contraste éclairant :
  - phi=1 « même percentile » : l'agent à destination occupe le MÊME percentile qu'à l'origine
    (hypothèse du premier Pi) ; optimiste vers un pays plus riche (on saute à son percentile).
  - role_ratio=1 « même rémunération réelle comparable » : l'agent gagne le même réel qu'en
    France et atterrit donc PLUS BAS dans une distribution de destination plus riche.
L'écart entre les deux mesure ce que phi=1 supposait. La friction reste au bloc D8 (kappa).
"""
from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel

from src.engine.patrimony import Patrimony


class RankTranslation(BaseModel):
    """Atterrissage à rôle comparable et gain patrimonial associé, à une position d'origine."""
    profile: str
    position: str
    origin: str
    destination: str
    role_ratio_ppp: float
    target_real_pp: float            # rémunération réelle visée à destination ($ int.)
    pi_origin_ce: float
    pi_destination_ce_landing: float  # Pi de destination au percentile d'atterrissage (interpolé)
    leave_gain_ce: float
    leave_dominates_ce: bool


class BreakEven(BaseModel):
    """Seuil de bascule role_ratio* : prime à rôle comparable qui égalise rester et partir.

    role_ratio_star est le ratio de rémunération réelle à rôle comparable (destination/origine,
    net du niveau) à partir duquel partir domine rester. None = la destination ne rattrape pas
    l'origine même à son sommet observé (partir ne paie à aucune prime plausible sur la plage).
    """
    profile: str
    position: str
    origin: str
    destination: str
    pi_origin_ce: float
    role_ratio_star: float | None


def break_even_role_ratio(origin: List[Patrimony],
                          destination: List[Patrimony]) -> List[BreakEven]:
    """role_ratio* par (profil, position) : seuil où Pi destination (atterrissage) = Pi origine."""
    out: List[BreakEven] = []
    for profile in sorted({p.profile for p in origin}):
        dest = sorted([p for p in destination if p.profile == profile],
                      key=lambda p: p.real_pp_intl_usd)
        if not dest:
            continue
        xs = np.array([p.real_pp_intl_usd for p in dest])   # revenu réel de destination
        ys = np.array([p.pi_ce for p in dest])              # Pi correspondant (croissant)
        for o in [p for p in origin if p.profile == profile]:
            if o.pi_ce >= ys[-1]:
                star = None                                  # destination ne rattrape pas, même au sommet
            else:
                target_real = float(np.interp(o.pi_ce, ys, xs))  # revenu réel où Pi_dest = Pi_origine
                star = round(target_real / o.real_pp_intl_usd, 3)
            out.append(BreakEven(
                profile=profile, position=o.position, origin=o.country,
                destination=dest[0].country, pi_origin_ce=o.pi_ce, role_ratio_star=star,
            ))
    return out


def translate_rank(origin: List[Patrimony], destination: List[Patrimony],
                   role_ratio_ppp: float) -> List[RankTranslation]:
    """Évalue Pi de destination au percentile d'atterrissage à rôle comparable, par profil."""
    out: List[RankTranslation] = []
    for profile in sorted({p.profile for p in origin}):
        dest = sorted([p for p in destination if p.profile == profile],
                      key=lambda p: p.real_pp_intl_usd)
        if not dest:
            continue
        xs = np.array([p.real_pp_intl_usd for p in dest])
        ys = np.array([p.pi_ce for p in dest])
        for o in [p for p in origin if p.profile == profile]:
            target = o.real_pp_intl_usd * role_ratio_ppp
            pi_land = float(np.interp(target, xs, ys))  # extrapole en plateau hors bornes
            gain = round(pi_land - o.pi_ce)
            out.append(RankTranslation(
                profile=profile, position=o.position,
                origin=o.country, destination=dest[0].country,
                role_ratio_ppp=role_ratio_ppp, target_real_pp=round(target, 1),
                pi_origin_ce=o.pi_ce, pi_destination_ce_landing=round(pi_land),
                leave_gain_ce=gain, leave_dominates_ce=gain > 0,
            ))
    return out
