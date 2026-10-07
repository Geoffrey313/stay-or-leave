"""Paramètres de la dynamique (Bellman), lus depuis calibration.csv (hypothèses déclarées)."""
from __future__ import annotations

import csv

from pydantic import BaseModel

from src.common.paths import param_file


class Dynamics(BaseModel):
    """Paramètres du modèle dynamique (tous [I]/[E], cadre)."""
    sigma_eta: float          # choc permanent (marche aléatoire)
    kappa: float              # coût fixe de migration ($ int.)
    amenity: float            # aménité chez-soi, flux annuel ($ int.)
    lambda_return: float      # coût de retour / coût de départ, à durée nulle
    return_cost_slope: float  # delta : pente du coût de retour en fraction de carrière écoulée
    age_start: int
    age_retire: int
    age_mig_max: int


def load_dynamics() -> Dynamics:
    with param_file("calibration.csv").open(newline="") as fh:
        d = {r["param"]: r["value"] for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))}
    return Dynamics(
        sigma_eta=float(d["sigma_eta_permanent"]),
        kappa=float(d["migration_cost_kappa_intl_usd"]),
        amenity=float(d["amenity_A_intl_usd"]),
        lambda_return=float(d["lambda_return"]),
        return_cost_slope=float(d["return_cost_duration_slope"]),
        age_start=int(float(d["age_start"])), age_retire=int(float(d["age_retire"])),
        age_mig_max=int(float(d["age_mig_max"])),
    )
