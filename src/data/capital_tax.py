"""Lecture de la fiscalité du capital (bloc D6) : taux d'imposition des plus-values.

Entrée : data/params/capital_tax.csv. Sert à capitaliser le surplus d'épargne libre net de
fiscalité du capital (rendement net = r_mkt * (1 - tau)).
"""
from __future__ import annotations

import csv
from typing import Dict

from pydantic import BaseModel, Field

from src.common.paths import param_file


class CapitalTax(BaseModel):
    """Taux d'imposition des plus-values mobilières d'un pays."""
    country: str
    capgains_tax_rate: float = Field(ge=0.0, lt=1.0)


def load_capital_tax() -> Dict[str, CapitalTax]:
    path = param_file("capital_tax.csv")
    with path.open(newline="") as fh:
        rows = [CapitalTax(country=r["country"], capgains_tax_rate=float(r["capgains_tax_rate"]))
                for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))]
    return {r.country: r for r in rows}
