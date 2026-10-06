"""Lecture de la prime salariale phi (bloc D7) : ratio de rémunération à rôle comparable.

Entrée : data/params/wage_premium.csv. role_ratio_ppp est une translation de RANG (nette du
niveau déjà ancré par la distribution), pas un facteur de niveau. Voir src/engine/wage_premium.py.
"""
from __future__ import annotations

import csv
from typing import Dict

from pydantic import BaseModel, PositiveFloat

from src.common.paths import param_file


class WagePremium(BaseModel):
    """Ratio de rémunération à rôle comparable (réel), destination vs France, net du niveau."""
    destination: str
    role_ratio_ppp: PositiveFloat


def load_wage_premium() -> Dict[str, WagePremium]:
    path = param_file("wage_premium.csv")
    with path.open(newline="") as fh:
        rows = [WagePremium(destination=r["destination"], role_ratio_ppp=float(r["role_ratio_ppp"]))
                for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))]
    return {r.destination: r for r in rows}


def load_occupation_premium() -> Dict[str, Dict[str, float]]:
    """Prime occupationnelle (groupe/moyenne nationale) par pays, pour le test du sommet.

    Retourne {pays: {occupation: premium_over_national_avg}}. role_ratio = premium_dest/premium_origine.
    """
    path = param_file("occupation_premium.csv")
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh)):
            out.setdefault(r["country"], {})[r["occupation"]] = float(r["premium_over_national_avg"])
    return out


def load_education_premium() -> Dict[str, Dict[str, float]]:
    """Gains relatifs par niveau de diplôme et pays (OCDE EaG), pour la sensibilité D7.

    Retourne {pays: {attainment: relative_earnings}}. role_ratio = premium_dest/premium_origine.
    """
    path = param_file("education_premium.csv")
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh)):
            out.setdefault(r["country"], {})[r["attainment"]] = float(r["relative_earnings"])
    return out
