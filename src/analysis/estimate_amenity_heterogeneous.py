"""Estimation de l'attachement au pays HÉTÉROGÈNE : distribution de A sur la cohorte (bloc D13).

La SMM à A représentatif (estimate_amenity_migration) a montré que les corridors fortement
attractifs (CH : bascule + niveau uniforme) sont QUASI-BIMODAUX : le stock net saute de ~1 à ~0
sans passer par la valeur observée. Un agent représentatif ENCADRE mais n'atteint pas
l'intermédiaire. La correction déclarée : l'attachement A n'est pas un scalaire commun mais
hétérogène entre individus, A_i ~ loi de moyenne mu_A et dispersion sigma_A. Le stock net de la
cohorte devient alors LISSE : part des individus dont A est sous le seuil de migration.

Clé de calcul : net(A) par agent ne dépend PAS de la distribution. On pré-calcule donc net(A) (et
les comptes de départs/retours par âge) sur une GRILLE de A, UNE fois (quelques résolutions), puis
la recherche sur (mu_A, sigma_A) n'est qu'une re-pondération des résultats pré-calculés -> rapide.
kappa est fixé (optimum du run homogène) pour isoler l'effet de l'hétérogénéité ; déclaré.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from src.data.cost_of_living import load_cost_of_living
from src.data.pension import load_pension, load_calibration
from src.data.capital_tax import load_capital_tax
from src.data.housing import load_housing
from src.data.dynamics import load_dynamics
from src.engine.bellman import solve
from src.engine.simulate import simulate_moments
from src.analysis.estimate_amenity_migration import load_observed


def _trunc_normal_weights(a_grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Poids d'une Normale(mu, sigma) tronquée à >=0, discrétisée sur a_grid (normalisée)."""
    w = np.exp(-0.5 * ((a_grid - mu) / sigma) ** 2)
    w[a_grid < 0] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.ones_like(a_grid) / len(a_grid)


def precompute_net_by_amenity(destination: str, kappa: float, role_ratio=None,
                              a_grid=None) -> Dict[float, dict]:
    """net(A) et comptes bruts par âge, pour A sur la grille (une résolution Bellman par A)."""
    col, pen, tax, house = load_cost_of_living(), load_pension(), load_capital_tax(), load_housing()
    cal, dyn = load_calibration(), load_dynamics()
    a_grid = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000]) if a_grid is None else np.asarray(a_grid)
    out = {}
    for A in a_grid:
        d = dyn.model_copy(update={"amenity": float(A), "kappa": float(kappa)})
        sim = simulate_moments(solve("fr", destination, col, pen, tax, house, cal, d, role_ratio=role_ratio))
        total_dep = sim["emigrant_share"]
        out[float(A)] = {"net": sim["abroad_share_end"], "total_dep": total_dep,
                         "total_ret": sim["return_rate"] * total_dep,
                         "dep_by_age": sim["departures_by_age"]}
    return out


def _cohort_moments(by_a: Dict[float, dict], a_grid: np.ndarray, w: np.ndarray) -> dict:
    """Moments de la cohorte = moyenne pondérée (par la distribution de A) des comptes par A."""
    net = sum(wi * by_a[float(A)]["net"] for A, wi in zip(a_grid, w))
    tot_dep = sum(wi * by_a[float(A)]["total_dep"] for A, wi in zip(a_grid, w))
    tot_ret = sum(wi * by_a[float(A)]["total_ret"] for A, wi in zip(a_grid, w))
    ages = sorted(next(iter(by_a.values()))["dep_by_age"].keys())
    dep_age = {t: sum(wi * by_a[float(A)]["dep_by_age"][t] for A, wi in zip(a_grid, w)) for t in ages}
    peak = max(dep_age, key=lambda t: dep_age[t]) if tot_dep > 0 else None
    return {"abroad_share_end": net, "emigrant_share": tot_dep,
            "return_rate": (tot_ret / tot_dep) if tot_dep > 0 else 0.0, "peak_departure_age": peak}


def _distance(sim: dict, obs: Dict[str, float]) -> float:
    d = 0.0
    if sim["peak_departure_age"] is not None:
        d += ((sim["peak_departure_age"] - obs["peak_departure_age"]) / obs["peak_departure_age"]) ** 2
    else:
        d += 1.0
    d += ((sim["abroad_share_end"] - obs["emigrant_share"]) / max(obs["emigrant_share"], 1e-6)) ** 2
    d += 0.25 * ((sim["return_rate"] - obs["return_rate"]) / max(obs["return_rate"], 1e-6)) ** 2
    return d


def estimate_heterogeneous(destination="ch", moments_csv="migration_moments_ch.csv",
                           kappa=20000.0, role_ratio=None, mu_grid=None, sigma_grid=None,
                           a_grid=None) -> dict:
    """Estime (mu_A, sigma_A) de la distribution d'attachement qui reproduit le flux du corridor.
    net(A) pré-calculé une fois, puis re-pondéré -> la recherche (mu, sigma) est rapide."""
    a_grid = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000]) if a_grid is None else np.asarray(a_grid)
    by_a = precompute_net_by_amenity(destination, kappa, role_ratio, a_grid)
    obs = load_observed(moments_csv)
    mu_grid = np.arange(0, 20001, 2500) if mu_grid is None else np.asarray(mu_grid)
    sigma_grid = np.array([2000, 4000, 6000, 8000, 10000, 15000]) if sigma_grid is None else np.asarray(sigma_grid)
    best = None
    for mu in mu_grid:
        for sigma in sigma_grid:
            w = _trunc_normal_weights(a_grid, float(mu), float(sigma))
            sim = _cohort_moments(by_a, a_grid, w)
            dist = _distance(sim, obs)
            if best is None or dist < best["distance"]:
                best = {"mu_A": float(mu), "sigma_A": float(sigma), "distance": dist,
                        "moments_sim": sim, "moments_obs": obs}
    best["net_by_amenity"] = {float(A): by_a[float(A)]["net"] for A in a_grid}
    best["kappa"] = kappa
    return best


if __name__ == "__main__":
    for dest, mom in [("ch", "migration_moments_ch.csv"), ("us", "migration_moments.csv")]:
        r = estimate_heterogeneous(destination=dest, moments_csv=mom)
        s = r["moments_sim"]; cref = 30957
        print(f"FR->{dest} hétérogène : mu_A={r['mu_A']:.0f} ({r['mu_A']/cref:.2f} c_ref) sigma_A={r['sigma_A']:.0f}"
              f" dist={r['distance']:.3f} ; net={s['abroad_share_end']:.3f} vs {r['moments_obs']['emigrant_share']} obs,"
              f" retour={s['return_rate']:.3f} pic={s['peak_departure_age']}")
