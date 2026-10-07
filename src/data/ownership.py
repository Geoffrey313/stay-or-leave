"""Lecture des paramètres du canal propriété (bloc D4), depuis data/params/ownership.csv.

Format long (country,param,value,...). Le logement de référence (taille ref_size_m2) est commun
aux pays ; son prix par pays = price_m2_local * ref_size_m2. Voir src/engine/housing_channel.py
pour la mécanique (accession sous DTI, équité, loyer imputé différentiel, vente exonérée à frais).
"""
from __future__ import annotations

import csv
from typing import Dict, Tuple

from pydantic import BaseModel

from src.common.paths import param_file


class OwnershipParams(BaseModel):
    """Paramètres d'accession et de propriété d'un pays (monnaie locale)."""
    country: str
    price_m2_local: float
    mortgage_rate: float
    dti_max: float
    dti_is_lti: bool           # True = le plafond est un loan-to-income (UK), DTI posé en proxy
    term_years: int
    down_payment_share: float
    sale_fee_share: float      # frais de transaction côté VENDEUR (vente à la migration)

    def reference_price(self, ref_size_m2: float) -> float:
        """Prix du logement de référence en monnaie locale."""
        return self.price_m2_local * ref_size_m2


def load_ownership() -> Tuple[Dict[str, OwnershipParams], float]:
    """Renvoie ({pays: OwnershipParams}, ref_size_m2)."""
    rows = {}
    ref_size = None
    with param_file("ownership.csv").open(newline="") as fh:
        for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh)):
            if r["param"] == "ref_size_m2":
                ref_size = float(r["value"])
                continue
            rows.setdefault(r["country"], {})[r["param"]] = r["value"]
    params = {c: OwnershipParams(
        country=c,
        price_m2_local=float(d["price_m2_local"]),
        mortgage_rate=float(d["mortgage_rate"]),
        dti_max=float(d["dti_max"]),
        dti_is_lti=bool(int(d["dti_is_lti"])),
        term_years=int(float(d["term_years"])),
        down_payment_share=float(d["down_payment_share"]),
        sale_fee_share=float(d["sale_fee_share"]),
    ) for c, d in rows.items()}
    if ref_size is None:
        raise ValueError("ref_size_m2 manquant dans ownership.csv")
    return params, ref_size
