"""Lecture des cascades socio-fiscales par pays (schéma pays-neutre).

Entrée (produite hors dépôt, voir data-acquisition/) : data/params/<country>/cascade.csv.
Ce module lit et valide ; le calcul des taux de conversion est dans src/engine. Sert le
bloc D1 (cascade coût employeur -> revenu disponible, hors taxe sur la consommation : la
TVA / sales tax relève du bloc D3, indice IC_p).
"""
from __future__ import annotations

import csv
from typing import List

from pydantic import BaseModel, NonNegativeInt

from src.common.paths import param_file


class CascadePoint(BaseModel):
    """Composantes de la cascade d'un pays, au coût employeur (monnaie locale)."""
    country: str
    currency: str
    profile: str
    cost_employer: NonNegativeInt
    benefits_year: NonNegativeInt
    disposable_year: NonNegativeInt


def available_countries() -> List[str]:
    """Pays dont une cascade est livrée, triés."""
    base = param_file()  # data/params
    return sorted(d.name for d in base.iterdir() if (d / "cascade.csv").exists())


def load_cascade(country: str) -> List[CascadePoint]:
    """Charge la cascade d'un pays, triée par profil puis coût employeur croissant."""
    path = param_file(country, "cascade.csv")
    with path.open(newline="") as fh:
        rows = [CascadePoint(**r) for r in csv.DictReader(fh)]
    return sorted(rows, key=lambda p: (p.profile, p.cost_employer))
