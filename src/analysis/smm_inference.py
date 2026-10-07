"""Inférence robuste à l'identification pour la méthode des moments simulés (bloc SL-US-04).

Les estimés d'aménité `A` et de coût de migration `kappa` sont rapportés avec un JEU DE CONFIANCE,
pas un écart-type par delta-méthode : `A` est faiblement identifiée (estimation centrale au coin
pour plusieurs corridors), si bien qu'une variance asymptotique symétrique serait trompeuse. Le jeu
de confiance est obtenu en INVERSANT l'objectif : la région des paramètres dont les moments simulés
tombent tous dans les bandes observées déclarées (`BANDS`), bandes qui reflètent la nature grossière
[E] des moments de flux. Déterministe ; on réutilise la grille de la MMS, aucune modification du cœur.

Ce module apporte l'INFÉRENCE (point 3 de l'audit). Il ne source pas la prime d'élite (point 1) et
n'identifie pas `eta` indépendamment de sa cible (point 2) : le jeu de confiance de `A` peut inclure
le coin, ce qui est précisément le constat du point 1, rapporté honnêtement.
"""
from __future__ import annotations

from typing import Dict

from src.common.paths import param_file  # noqa: F401  (cohérence d'import avec le reste de l'analyse)
from src.data.cost_of_living import load_cost_of_living
from src.data.pension import load_pension, load_calibration
from src.data.capital_tax import load_capital_tax
from src.data.housing import load_housing
from src.data.dynamics import load_dynamics
from src.engine.bellman import solve
from src.engine.simulate import simulate_moments
from src.analysis.estimate_amenity_migration import (load_observed, _distance, _sim_moment_map,
                                                     BANDS)


def _in_bands(mmap: Dict[str, float], obs: Dict[str, float]) -> bool:
    """Vrai si chaque moment simulé est dans la bande observée déclarée (jeu de confiance)."""
    for key, band in BANDS.items():
        if key not in obs:
            continue
        sim_v = mmap[key]
        if sim_v is None:                      # pic indéfini (aucun départ) : hors bande
            return False
        if abs(sim_v - obs[key]) > band:
            return False
    return True


def confidence_set(destination: str = "us", role_ratio: float | None = None,
                   moments_csv: str = "migration_moments.csv",
                   a_grid=None, kappa_grid=None, ownership: bool = False) -> dict:
    """Jeu de confiance robuste à l'identification pour (A, kappa), corridor FR -> destination.

    Balaye la grille (A, kappa), simule les moments, et retient les points dont TOUS les moments
    tombent dans les bandes observées. Renvoie le point (distance minimale), le jeu, l'étendue de A
    et de kappa sur le jeu, la présence du coin A=0, et l'étendue de la part nette sur le jeu.
    """
    col, pen, tax, house = load_cost_of_living(), load_pension(), load_capital_tax(), load_housing()
    cal, dyn = load_calibration(), load_dynamics()
    obs = load_observed(moments_csv)
    a_grid = ([0, 2500, 5000, 7500, 10000, 12500] if destination != "ch"
              else [0, 5000, 10000, 12000, 13000, 14000, 15000, 20000, 30000]) if a_grid is None else a_grid
    kappa_grid = [10000, 20000, 35000, 50000, 75000, 100000] if kappa_grid is None else kappa_grid
    in_set, best, nets = [], None, []
    for A in a_grid:
        for kap in kappa_grid:
            d = dyn.model_copy(update={"amenity": float(A), "kappa": float(kap)})
            sim = simulate_moments(solve("fr", destination, col, pen, tax, house, cal, d,
                                         role_ratio=role_ratio, ownership=ownership))
            dist = _distance(sim, obs)
            if best is None or dist < best[2]:
                best = (A, kap, dist, sim["abroad_share_end"])
            if _in_bands(_sim_moment_map(sim), obs):
                in_set.append((A, kap, dist, sim["abroad_share_end"]))
                nets.append(sim["abroad_share_end"])
    a_vals = sorted({p[0] for p in in_set}); k_vals = sorted({p[1] for p in in_set})
    return {"destination": destination, "role_ratio": role_ratio,
            "best_A": best[0], "best_kappa": best[1], "best_distance": best[2],
            "n_in_set": len(in_set),
            "A_range": (a_vals[0], a_vals[-1]) if a_vals else None,
            "kappa_range": (k_vals[0], k_vals[-1]) if k_vals else None,
            "corner_in_set": any(p[0] == 0 for p in in_set),
            "net_range": (min(nets), max(nets)) if nets else None,
            "set": in_set}


def confidence_set_heterogeneous(destination: str = "ch", moments_csv: str = "migration_moments_ch.csv",
                                 kappa: float = 20000.0, role_ratio: float | None = None,
                                 dest_appeal: float = 0.0, mu_grid=None, sigma_grid=None) -> dict:
    """Jeu de confiance robuste à l'identification pour l'estimand HÉTÉROGÈNE (mu_A, sigma_A), pour les
    corridors quasi-bimodaux où l'agent représentatif ne peut pas reproduire le niveau observé. On
    pré-calcule net(A) une fois, puis on balaye (mu_A, sigma_A) par re-pondération (rapide) et on
    retient les points dont les moments tombent dans les bandes observées."""
    import numpy as np
    from src.analysis.estimate_amenity_heterogeneous import (precompute_net_by_amenity, _cohort_moments,
                                                             _trunc_normal_weights, _distance as _dist_h)
    obs = load_observed(moments_csv)
    a_grid = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000])
    by_a = precompute_net_by_amenity(destination, kappa, role_ratio, a_grid, dest_appeal)
    mu_grid = np.arange(0, 20001, 2500) if mu_grid is None else np.asarray(mu_grid)
    sigma_grid = np.array([2000, 4000, 6000, 8000, 10000, 15000]) if sigma_grid is None else np.asarray(sigma_grid)
    in_set, best, nets = [], None, []
    for mu in mu_grid:
        for sigma in sigma_grid:
            w = _trunc_normal_weights(a_grid, float(mu), float(sigma))
            sim = _cohort_moments(by_a, a_grid, w)
            dist = _dist_h(sim, obs)
            if best is None or dist < best[2]:
                best = (float(mu), float(sigma), dist, sim["abroad_share_end"])
            mmap = {"emigrant_share": sim["abroad_share_end"], "peak_departure_age": sim["peak_departure_age"],
                    "return_rate": sim["return_rate"]}
            if _in_bands(mmap, obs):
                in_set.append((float(mu), float(sigma), dist, sim["abroad_share_end"]))
                nets.append(sim["abroad_share_end"])
    mu_vals = sorted({p[0] for p in in_set}); s_vals = sorted({p[1] for p in in_set})
    return {"destination": destination, "best_mu": best[0], "best_sigma": best[1], "best_distance": best[2],
            "n_in_set": len(in_set),
            "mu_range": (mu_vals[0], mu_vals[-1]) if mu_vals else None,
            "sigma_range": (s_vals[0], s_vals[-1]) if s_vals else None,
            "net_range": (min(nets), max(nets)) if nets else None, "set": in_set}


if __name__ == "__main__":
    for rr, lab in [(None, "proxy sourcé"), (1.27, "élite médiane")]:
        cs = confidence_set(role_ratio=rr)
        print(f"FR->US ({lab}) : A*={cs['best_A']:.0f} kappa*={cs['best_kappa']:.0f} ; "
              f"jeu n={cs['n_in_set']} A in {cs['A_range']} kappa in {cs['kappa_range']} ; "
              f"coin A=0 dans le jeu = {cs['corner_in_set']} ; net sur le jeu = {cs['net_range']}")
