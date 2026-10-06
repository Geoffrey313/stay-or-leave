"""Lecture de l'immobilier (bloc D4), baseline locataire.

Entrée : data/params/housing.csv (loyer mensuel de référence, monnaie locale). Le loyer est
une dépense pure soustraite du pouvoir d'achat réel avant le partage consommation/surplus.
Le canal propriété (épargne forcée, actif terminal) viendra ensuite.
"""
from __future__ import annotations

import csv
from typing import Dict

from pydantic import BaseModel, NonNegativeFloat

from src.common.paths import param_file


class Housing(BaseModel):
    """Loyer mensuel de référence d'un pays (monnaie locale)."""
    country: str
    rent_monthly_local: NonNegativeFloat


def load_housing() -> Dict[str, Housing]:
    path = param_file("housing.csv")
    with path.open(newline="") as fh:
        rows = [Housing(country=r["country"], rent_monthly_local=float(r["rent_monthly_local"]))
                for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))]
    return {r.country: r for r in rows}
