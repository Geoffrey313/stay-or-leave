"""Lecture du paramètre transformé France : taux de conversion theta_FR(C).

Entrée : data/params/fr/theta_fr.csv (produit hors dépôt par OpenFisca-France, voir
data-acquisition/build_theta_fr.py). Ce module ne calcule rien : il lit et valide.
Sert le bloc D1 (cascade coût employeur -> pouvoir d'achat) du modèle.
"""
from __future__ import annotations

import csv
from typing import List

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from src.common.paths import param_file


class ThetaPoint(BaseModel):
    """Un point de la cascade France, au coût employeur annuel donné (euros)."""
    cost_employer_eur: NonNegativeInt
    salaire_de_base_month_eur: NonNegativeInt
    salaire_net_year_eur: NonNegativeInt
    revenu_disponible_year_eur: NonNegativeInt
    theta_disposable: NonNegativeFloat


def load_theta_fr() -> List[ThetaPoint]:
    """Charge la cascade France, triée par coût employeur croissant."""
    path = param_file("fr", "theta_fr.csv")
    with path.open(newline="") as fh:
        rows = [ThetaPoint(**{k: v for k, v in r.items()}) for r in csv.DictReader(fh)]
    return sorted(rows, key=lambda p: p.cost_employer_eur)
