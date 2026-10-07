"""Estimation JOINTE multi-corridor : test de cohérence de l'attachement (bloc D13).

L'hétérogénéité par corridor a donné des moyennes d'attachement DIFFÉRENTES (FR->CH ~0,65 c_ref,
FR->US ~0), alors que l'attache à la France est UNE propriété des Français. Test de cohérence :
peut-on reproduire les DEUX flux avec UNE SEULE distribution d'attachement commune ?

V1 (attache commune seule) : une seule Normale(mu, sigma) tronquée, aucun terme par destination.
On minimise la distance combinée sur le stock net des deux corridors. Si le fit est mauvais, c'est
la TENSION nommée : avec les tirs (cascades, retraites, PPP) SOURCÉS, une attache commune ne peut
pas reproduire les deux flux -> A n'est pas une attache universelle.

V2 (commune + proximité par destination) : même distribution d'attachement à la France, PLUS un
terme delta_dest par destination (appel propre au pays : proximité, langue, réseaux), delta_US=0 en
référence. L'amenity effectif du corridor dest est A_i - delta_dest (la proximité érode le maintien).
delta_CH qui réconcilie les deux flux EST l'estimande de l'appel spécifique à la Suisse.

net(A) par corridor est pré-calculé une fois (estimate_amenity_heterogeneous) ; le joint n'est que de
la re-pondération / un décalage -> rapide.
"""
from __future__ import annotations

import numpy as np

from src.analysis.estimate_amenity_heterogeneous import (precompute_net_by_amenity,
                                                         _trunc_normal_weights)
from src.analysis.estimate_amenity_migration import load_observed

CORRIDORS = {"ch": "migration_moments_ch.csv", "us": "migration_moments.csv"}


def _net_arrays(kappa=20000.0, a_grid=None):
    """net(A) par corridor sur une grille A commune (une résolution Bellman par (corridor, A))."""
    a_grid = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000]) if a_grid is None else np.asarray(a_grid)
    rr = {"ch": 1.0, "us": None}  # CH : story = bascule (role_ratio 1) ; US : proxy sourcé
    net = {}
    for dest in CORRIDORS:
        by_a = precompute_net_by_amenity(dest, kappa, rr[dest], a_grid)
        net[dest] = np.array([by_a[float(A)]["net"] for A in a_grid])
    return a_grid, net


def estimate_joint(kappa=20000.0, mu_grid=None, sigma_grid=None, delta_grid=None, a_grid=None) -> dict:
    a_grid, net = _net_arrays(kappa, a_grid)
    obs = {d: load_observed(CORRIDORS[d])["emigrant_share"] for d in CORRIDORS}
    mu_grid = np.arange(0, 25001, 1250) if mu_grid is None else np.asarray(mu_grid)
    sigma_grid = np.array([2000, 4000, 6000, 8000, 10000, 15000]) if sigma_grid is None else np.asarray(sigma_grid)
    delta_grid = np.arange(0, 25001, 1250) if delta_grid is None else np.asarray(delta_grid)

    def cohort_net(dest, mu, sigma, delta=0.0):
        w = _trunc_normal_weights(a_grid, mu, sigma)
        # amenity effectif du corridor = A + delta_dest : delta_dest>0 = attache-équivalente
        # SUPPLÉMENTAIRE que la destination exige (son appel/pull propre, au-delà du commun).
        # net évalué en A+delta. delta_US=0 (référence) -> mu capte l'attache commune (terme US).
        n = np.interp(a_grid + delta, a_grid, net[dest])
        return float((w * n).sum())

    def reldist(dest, val):
        return ((val - obs[dest]) / max(obs[dest], 1e-6)) ** 2

    # V1 : attache commune seule (delta = 0 partout)
    v1 = None
    for mu in mu_grid:
        for sigma in sigma_grid:
            nets = {d: cohort_net(d, float(mu), float(sigma)) for d in CORRIDORS}
            dist = sum(reldist(d, nets[d]) for d in CORRIDORS)
            if v1 is None or dist < v1["distance"]:
                v1 = {"mu_A": float(mu), "sigma_A": float(sigma), "distance": dist, "net": nets}

    # V2 : commune + delta_CH (delta_US = 0 en référence)
    v2 = None
    for mu in mu_grid:
        for sigma in sigma_grid:
            for dCH in delta_grid:
                nets = {"us": cohort_net("us", float(mu), float(sigma), 0.0),
                        "ch": cohort_net("ch", float(mu), float(sigma), float(dCH))}
                dist = sum(reldist(d, nets[d]) for d in CORRIDORS)
                if v2 is None or dist < v2["distance"]:
                    v2 = {"mu_A": float(mu), "sigma_A": float(sigma), "delta_CH": float(dCH),
                          "distance": dist, "net": nets}
    return {"obs": obs, "v1_common_only": v1, "v2_common_plus_proximity": v2, "kappa": kappa}


if __name__ == "__main__":
    r = estimate_joint()
    cref = 30957
    o = r["obs"]
    print(f"Observé : net CH={o['ch']} US={o['us']}")
    v1 = r["v1_common_only"]
    print(f"\nV1 (attache COMMUNE seule) : mu_A={v1['mu_A']:.0f} ({v1['mu_A']/cref:.2f} c_ref) sigma_A={v1['sigma_A']:.0f}")
    print(f"   -> net CH={v1['net']['ch']:.3f} US={v1['net']['us']:.3f} ; distance combinée {v1['distance']:.3f}")
    v2 = r["v2_common_plus_proximity"]
    print(f"\nV2 (commune + proximité delta_CH, delta_US=0) : mu_A={v2['mu_A']:.0f} ({v2['mu_A']/cref:.2f} c_ref)"
          f" sigma_A={v2['sigma_A']:.0f} delta_CH={v2['delta_CH']:.0f} ({v2['delta_CH']/cref:.2f} c_ref)")
    print(f"   -> net CH={v2['net']['ch']:.3f} US={v2['net']['us']:.3f} ; distance combinée {v2['distance']:.3f}")
