"""Lecture des composantes de la cascade France.

Entrée (produite hors dépôt, voir data-acquisition/build_theta_fr.py) :
  - data/params/fr/theta_fr.csv : composantes brutes par coût employeur.
Ce module lit et valide ; le calcul des taux de conversion est dans src/engine.
Sert le bloc D1 (cascade coût employeur -> revenu disponible, hors TVA). La taxe sur la
consommation relève du bloc D3 (indice IC_p) et n'entre pas ici.
"""
from __future__ import annotations

import csv
from typing import List

from pydantic import BaseModel, NonNegativeInt

from src.common.paths import param_file


class CascadePoint(BaseModel):
    """Composantes brutes de la cascade France au coût employeur annuel (euros)."""
    profile: str
    cost_employer_eur: NonNegativeInt
    salaire_de_base_month_eur: NonNegativeInt
    salaire_net_year_eur: NonNegativeInt
    prestations_sociales_year_eur: NonNegativeInt
    revenu_disponible_year_eur: NonNegativeInt


def load_cascade_fr() -> List[CascadePoint]:
    """Charge les composantes France, triées par profil puis coût employeur croissant."""
    path = param_file("fr", "theta_fr.csv")
    with path.open(newline="") as fh:
        rows = [CascadePoint(**r) for r in csv.DictReader(fh)]
    return sorted(rows, key=lambda p: (p.profile, p.cost_employer_eur))
