"""Estimation de kappa (coût de migration) et A (aménité) par méthode des moments simulés (D13).

Minimise la distance entre moments simulés (simulate_moments sur la Bellman re-résolue pour
chaque candidat) et moments observés des flux (data/params/migration_moments.csv) :
    (A*, kappa*) = argmin  sum_m w_m * ((m_sim - m_obs)/m_obs)^2.
Moments : pic de départ par âge (identifie surtout kappa via l'amortissement sur l'horizon),
part d'émigrés (niveau conjoint A+kappa), taux de retour (valeur de l'option de réversibilité).
Identification grossière (moments agrégés) : on sort (A*, kappa*) ET une lecture de la surface,
pas un point sans incertitude. Si la surface est plate, repli déclaré : A/kappa assumés, flux
en validation (cf. estimation-design.md).
"""
from __future__ import annotations

import csv
from typing import Dict, Tuple

import numpy as np

from src.common.paths import param_file
from src.data.cost_of_living import load_cost_of_living
from src.data.pension import load_pension, load_calibration
from src.data.capital_tax import load_capital_tax
from src.data.housing import load_housing
from src.data.dynamics import load_dynamics
from src.engine.bellman import solve
from src.engine.simulate import simulate_moments


def load_observed() -> Dict[str, float]:
    with param_file("migration_moments.csv").open(newline="") as fh:
        return {r["moment"]: float(r["value"])
                for r in csv.DictReader(filter(lambda l: not l.startswith("#"), fh))}


def _distance(sim: dict, obs: Dict[str, float]) -> float:
    # Part d'émigrés = STOCK NET à l'étranger (bien défini) ; le taux de retour est peu fiable
    # en premier passage (churn de la simulation au W représentatif) -> poids réduit.
    d = 0.0
    if sim["peak_departure_age"] is not None:
        d += ((sim["peak_departure_age"] - obs["peak_departure_age"]) / obs["peak_departure_age"]) ** 2
    else:
        d += 1.0
    d += ((sim["abroad_share_end"] - obs["emigrant_share"]) / max(obs["emigrant_share"], 1e-6)) ** 2
    d += 0.25 * ((sim["return_rate"] - obs["return_rate"]) / max(obs["return_rate"], 1e-6)) ** 2
    return d


def estimate(a_grid=None, kappa_grid=None, role_ratio=None) -> dict:
    """Recherche sur grille (A, kappa) ; renvoie le meilleur et la surface.

    role_ratio : prime à rôle comparable D7 injectée dans l'atterrissage en rang. None = valeur
    sourcée (proxy éducation, wage_premium.csv) ; passer une valeur de queue (occupation d'élite)
    pour la sensibilité du régime où vivent les partants réels.
    """
    col, pen, tax, house = load_cost_of_living(), load_pension(), load_capital_tax(), load_housing()
    cal, dyn = load_calibration(), load_dynamics()
    obs = load_observed()
    # grille A recentrée sur la région informative (émigration positive) : sous l'utilité
    # commensurable, une aménité > ~0,4*c_ref éteint toute émigration (home trop attractif).
    a_grid = [0, 2500, 5000, 7500, 10000, 12500] if a_grid is None else a_grid
    kappa_grid = [10000, 20000, 35000, 50000, 75000, 100000] if kappa_grid is None else kappa_grid
    best, surface = None, []
    for A in a_grid:
        for kap in kappa_grid:
            d = dyn.model_copy(update={"amenity": float(A), "kappa": float(kap)})
            res = solve("fr", "us", col, pen, tax, house, cal, d, role_ratio=role_ratio)
            sim = simulate_moments(res)
            dist = _distance(sim, obs)
            surface.append((A, kap, dist, sim["peak_departure_age"], sim["emigrant_share"], sim["return_rate"]))
            if best is None or dist < best[2]:
                best = (A, kap, dist, sim)
    a_floor, a_ceil = a_grid[0], a_grid[-1]
    return {"best_A": best[0], "best_kappa": best[1], "distance": best[2],
            "moments_sim": best[3], "moments_obs": obs, "surface": surface,
            # une estimation au bord de la grille A n'est PAS un point intérieur : les flux disent
            # seulement « A aussi bas (ou haut) que permis », A n'est pas séparable de kappa là.
            "A_at_lower_bound": best[0] <= a_floor, "A_at_upper_bound": best[0] >= a_ceil}


# Ancres de role_ratio pour la courbe A*(role_ratio), TOUTES SOURCÉES désormais (flag levé) :
# 0,97 = proxy éducation tertiaire (OCDE EaG) ; 1,09 = prime d'élite US, MOYENNE ; 1,27 = prime
# d'élite US, MÉDIANE (Eurostat SES 2018 base FR + BLS OEWS 2023 base US, dirigeants = concept
# propre, cf. occupation_premium.csv). La borne US sourcée (1,07-1,27) STRADDLE le seuil A>0 : A=0
# sur la moyenne, A>0 (~0,24 c_ref) sur la médiane. Le UK sourcé est <= ~1 (A au plancher), le
# résultat d'aménité est un phénomène US. On publie A/kappa en FONCTION du role_ratio, pas un verdict.
ROLE_RATIO_ANCHORS = [(0.970, "US proxy éducation (OCDE EaG)"),
                      (1.090, "US élite, moyenne (SES/OEWS)"),
                      (1.270, "US élite dirigeants, médiane (SES/OEWS)")]


def estimate_amenity_curve(anchors=ROLE_RATIO_ANCHORS) -> list:
    """A*(role_ratio) et kappa*(role_ratio) sur les ancres : la bande de sensibilité qui situe
    l'incertitude. Chaque point est une SMM complète au role_ratio donné."""
    out = []
    for rr, kind in anchors:
        e = estimate(role_ratio=rr)
        out.append({"role_ratio": rr, "kind": kind, "A": e["best_A"], "kappa": e["best_kappa"],
                    "distance": e["distance"], "A_at_lower_bound": e["A_at_lower_bound"],
                    "A_at_upper_bound": e["A_at_upper_bound"], "moments_sim": e["moments_sim"]})
    return out


if __name__ == "__main__":
    r = estimate()
    print(f"A* = {r['best_A']:.0f} , kappa* = {r['best_kappa']:.0f}  (distance {r['distance']:.3f})")
    s = r["moments_sim"]
    print(f"moments simulés : pic âge={s['peak_departure_age']}, part NETTE à l'étranger="
          f"{s['abroad_share_end']:.3f}, taux retour={s['return_rate']:.3f} (churn, peu fiable)")
    print(f"moments observés: {r['moments_obs']}")
