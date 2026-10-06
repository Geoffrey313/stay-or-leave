"""Taux de conversion coût employeur -> revenu disponible (bloc D1 du modèle).

θ est la primitive de la cascade socio-fiscale, HORS taxe sur la consommation : la TVA ou
sales tax vit une seule fois, dans la couche coût de la vie (indice IC_p, bloc D3), où les
niveaux de prix comparatifs sont déjà taxe incluse. La mettre aussi dans θ la compterait
deux fois.

Deux objets, ceux de la Partie A du cadre :
  - θ brut   : conversion du salaire seul, prestations exclues ;
  - θ élargi : prestations sociales incluses (transferts et extinction sous condition).
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.data.cascade import CascadePoint


class ConversionPoint(BaseModel):
    """Taux de conversion au coût employeur donné (monnaie locale), hors taxe conso."""
    country: str
    profile: str
    position: str
    cost_employer: int
    theta_brut: float      # (disponible - prestations) / coût employeur
    theta_elargi: float    # disponible / coût employeur


def conversion(points: List[CascadePoint]) -> List[ConversionPoint]:
    """Forme θ brut et θ élargi à partir des composantes de la cascade d'un pays."""
    out: List[ConversionPoint] = []
    for p in points:
        c = p.cost_employer
        salaire_seul = p.disposable_year - p.benefits_year
        out.append(ConversionPoint(
            country=p.country,
            profile=p.profile,
            position=p.position,
            cost_employer=c,
            theta_brut=round(salaire_seul / c, 4),
            theta_elargi=round(p.disposable_year / c, 4),
        ))
    return out
