"""Lecture du coût de la vie (bloc D3) : parités de pouvoir d'achat sur la consommation.

Entrée : data/params/cost_of_living.csv (PPA consommation effective, TTC, monnaie locale par
dollar international). Sert à convertir un revenu disponible nominal local en pouvoir d'achat
réel comparable entre pays. La taxe sur la consommation est portée ici (panier TTC), pas dans θ.
"""
from __future__ import annotations

import csv
from typing import Dict

from pydantic import BaseModel, PositiveFloat

from src.common.paths import param_file


class CostOfLiving(BaseModel):
    """Parité de pouvoir d'achat d'un pays (monnaie locale par dollar international)."""
    country: str
    ppp_aic_local_per_intl_usd: PositiveFloat
    currency: str
    year: int


def load_cost_of_living() -> Dict[str, CostOfLiving]:
    """Charge les PPA par pays, indexées par code pays."""
    path = param_file("cost_of_living.csv")
    with path.open(newline="") as fh:
        rows = [CostOfLiving(**r) for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))]
    return {r.country: r for r in rows}
