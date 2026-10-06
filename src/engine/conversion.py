"""Taux de conversion coût employeur -> revenu disponible (bloc D1 du modèle).

θ est la primitive de la cascade socio-fiscale, HORS taxe sur la consommation : la TVA
vit une seule fois, dans la couche coût de la vie (indice IC_p, bloc D3), où les niveaux
de prix comparatifs sont déjà TVA incluse. La mettre aussi dans θ la compterait deux fois.

Deux objets, ceux de la Partie A du cadre :
  - θ brut   : conversion du salaire seul, prestations exclues ;
  - θ élargi : prestations sociales incluses (transferts et extinction sous condition).
Le salaire seul retire les prestations du revenu disponible ; θ élargi les réintègre, ce
qui fait apparaître la zone de transferts sous le salaire minimum.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.data.france_cascade import CascadePoint


class ConversionPoint(BaseModel):
    """Taux de conversion au coût employeur annuel donné (euros), hors TVA, par profil."""
    profile: str
    cost_employer_eur: int
    theta_brut: float      # salaire seul / coût employeur
    theta_elargi: float    # prestations incluses / coût employeur


def conversion_fr(points: List[CascadePoint]) -> List[ConversionPoint]:
    """Forme θ brut et θ élargi France à partir des composantes de la cascade."""
    out: List[ConversionPoint] = []
    for p in points:
        c = p.cost_employer_eur
        salaire_seul = p.revenu_disponible_year_eur - p.prestations_sociales_year_eur
        out.append(ConversionPoint(
            profile=p.profile,
            cost_employer_eur=c,
            theta_brut=round(salaire_seul / c, 4),
            theta_elargi=round(p.revenu_disponible_year_eur / c, 4),
        ))
    return out
