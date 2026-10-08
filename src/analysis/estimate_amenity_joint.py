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
from src.analysis.estimate_amenity_migration import load_observed, ROLE_RATIO_ANCHORS

CORRIDORS = {"ch": "migration_moments_ch.csv", "us": "migration_moments.csv",
             "uk": "migration_moments_uk.csv"}


def _net_arrays(kappa=20000.0, a_grid=None, us_role_ratio=None):
    """net(A) par corridor sur une grille A commune (une résolution Bellman par (corridor, A)).

    us_role_ratio : prime à rôle comparable du corridor US. None (défaut) -> proxy éducation sourcé
    wage_premium.role_ratio_ppp = 0,970 (résolu dans bellman.py) ; le défaut reproduit le joint
    historique au bit près. Une valeur explicite porte la prime occupationnelle sourcée (ancres
    1,09 moyenne / 1,27 médiane d'élite, cf. occupation_premium.csv) dans l'identification du joint."""
    a_grid = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000]) if a_grid is None else np.asarray(a_grid)
    rr = {"ch": 1.0, "us": us_role_ratio, "uk": None}  # CH bascule (role_ratio 1) ; UK proxy 0,789
    net = {}
    for dest in CORRIDORS:
        by_a = precompute_net_by_amenity(dest, kappa, rr[dest], a_grid)
        net[dest] = np.array([by_a[float(A)]["net"] for A in a_grid])
    return a_grid, net


def estimate_joint(kappa=20000.0, mu_grid=None, sigma_grid=None, delta_grid=None, a_grid=None,
                   us_role_ratio=None) -> dict:
    a_grid, net = _net_arrays(kappa, a_grid, us_role_ratio)
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

    # V2 : commune + proximité par destination (delta_US = 0 en référence) ; delta_CH ET delta_UK.
    # Le 3e corridor (UK) ajoute des contraintes -> identifie mieux l'attache commune vs spécifique.
    v2 = None
    for mu in mu_grid:
        for sigma in sigma_grid:
            us_net = cohort_net("us", float(mu), float(sigma), 0.0)
            for dCH in delta_grid:
                ch_net = cohort_net("ch", float(mu), float(sigma), float(dCH))
                for dUK in delta_grid:
                    nets = {"us": us_net, "ch": ch_net,
                            "uk": cohort_net("uk", float(mu), float(sigma), float(dUK))}
                    dist = sum(reldist(d, nets[d]) for d in CORRIDORS)
                    if v2 is None or dist < v2["distance"]:
                        v2 = {"mu_A": float(mu), "sigma_A": float(sigma), "delta_CH": float(dCH),
                              "delta_UK": float(dUK), "distance": dist, "net": nets}
    return {"obs": obs, "v1_common_only": v1, "v2_common_plus_proximity": v2, "kappa": kappa,
            "us_role_ratio": us_role_ratio}


def estimate_joint_band(anchors=ROLE_RATIO_ANCHORS, kappa=20000.0, mu_grid=None,
                        sigma_grid=None, delta_grid=None, a_grid=None) -> list:
    """Encadrement du joint sur la bande d'ancres de role_ratio du corridor US (P1, SL-US-05).

    Le joint tournait le corridor US sur le seul proxy éducation 0,970 ; on le ré-estime à chaque
    ancre sourcée (0,97 proxy, 1,09 élite moyenne, 1,27 élite médiane dirigeants ; occupation_premium.csv)
    et on rapporte (mu_A commun, sigma_A, delta_CH, delta_UK) EN FONCTION de la prime US, pas en un
    point. Changer la prime US déplace net(US), donc l'attache commune mu_A optimale, donc delta_UK :
    l'encadrement rend cette propagation lisible. Chaque ancre est une estimation jointe complète."""
    out = []
    for rr, label in anchors:
        r = estimate_joint(kappa=kappa, mu_grid=mu_grid, sigma_grid=sigma_grid,
                           delta_grid=delta_grid, a_grid=a_grid, us_role_ratio=rr)
        v2, obs = r["v2_common_plus_proximity"], r["obs"]
        # Décomposition du résidu par corridor (garde-fou 2 SL-US-05) : la distance combinée est
        # une somme de carrés d'écarts relatifs ; on l'expose corridor par corridor pour que le
        # lecteur voie que le UK n'est PAS calé (reldist_UK ~ 1,0) et que l'amélioration vient de
        # US+CH. Sans quoi une distance ~1,0 se lirait à tort comme un bon ajustement.
        resid = {d: ((v2["net"][d] - obs[d]) / max(obs[d], 1e-6)) ** 2 for d in CORRIDORS}
        out.append({"us_role_ratio": rr, "label": label, "mu_A": v2["mu_A"],
                    "sigma_A": v2["sigma_A"], "delta_CH": v2["delta_CH"],
                    "delta_UK": v2["delta_UK"], "distance": v2["distance"], "net": v2["net"],
                    "obs": obs, "resid": resid})
    return out


if __name__ == "__main__":
    r = estimate_joint()
    cref = 30957
    o = r["obs"]
    print(f"Observé : net CH={o['ch']} US={o['us']}")
    v1 = r["v1_common_only"]
    print(f"\nV1 (attache COMMUNE seule) : mu_A={v1['mu_A']:.0f} ({v1['mu_A']/cref:.2f} c_ref) sigma_A={v1['sigma_A']:.0f}")
    print(f"   -> net CH={v1['net']['ch']:.3f} US={v1['net']['us']:.3f} UK={v1['net']['uk']:.3f} ; distance combinée {v1['distance']:.3f}")
    v2 = r["v2_common_plus_proximity"]
    print(f"\nV2 (commune + proximité, delta_US=0) : mu_A={v2['mu_A']:.0f} ({v2['mu_A']/cref:.2f} c_ref)"
          f" sigma_A={v2['sigma_A']:.0f} delta_CH={v2['delta_CH']/cref:.2f} c_ref delta_UK={v2['delta_UK']/cref:.2f} c_ref")
    print(f"   -> net CH={v2['net']['ch']:.3f} US={v2['net']['us']:.3f} UK={v2['net']['uk']:.3f} ; distance {v2['distance']:.3f}")
