"""Simulation de cohorte sur la DISTRIBUTION JOINTE (position, W_f, W_p) -> moments de flux.

Corrige le churn du premier passage (qui simulait aux seuils au W représentatif) : on propage
la masse sur l'état complet via les grilles de politique du solveur. Comme partir coûte kappa
(baisse de W_f), le retour immédiat devient coûteux et l'oscillation départ/retour disparaît.
La masse qui bouge atterrit au RANG translaté (role_ratio, D7 : land_o2d en partant, land_d2o
en rentrant), pas au même indice ; le retour paie un coût CROISSANT avec la durée à l'étranger
(res.return_cost[t]). Transitions de patrimoine par snap au noeud le plus proche (déterministe) ;
transition de position par la matrice tridiagonale. Moments : taux de départ par âge (et pic),
part nette à l'étranger, taux de retour.
"""
from __future__ import annotations

import numpy as np


def _snap(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def simulate_moments(res) -> dict:
    """Moments simulés par propagation de la distribution jointe (position, W_f, W_p)."""
    o, d = res.origin, res.destination
    P = res.transition
    npos, nf, npp = len(res.positions), len(res.Wf_g), len(res.Wp_g)
    Wf, Wp = res.Wf_g, res.Wp_g
    # cartes de transition de patrimoine par (pays, position, cellule)
    nf_map = {c: np.array([[_snap(Wf, Wf[i] * (1 + res.r_f[c]) + res.flows[c][1][p]) for i in range(nf)]
                           for p in range(npos)]) for c in (o, d)}
    np_map = {c: np.array([[_snap(Wp, Wp[j] * (1 + res.r_p) + res.flows[c][2][p]) for j in range(npp)]
                           for p in range(npos)]) for c in (o, d)}

    D = {o: np.zeros((npos, nf, npp)), d: np.zeros((npos, nf, npp))}
    D[o][:, 0, 0] = res.pop_mass                        # entrée : masse de population par rang, patrimoine nul
    ages = res.ages
    departures, returns, origin_mass = {}, {}, {}
    for t in ages[:-1]:
        origin_mass[t] = D[o].sum()
        pl, pr = res.policy_leave.get(t), res.policy_return.get(t)
        dep = ret = 0.0
        if pl is not None and t <= res.age_mig_max:     # départs origine -> destination (coût kappa sur W_f)
            moved = D[o] * pl
            dep = moved.sum()
            if dep > 0:
                D[o] = D[o] - moved
                add = np.zeros_like(D[d])
                for p in range(npos):
                    pland = int(res.land_o2d[p])         # atterrissage en rang (D7)
                    for i in range(nf):
                        inew = _snap(Wf, max(Wf[i] - res.kappa, 0.0))
                        for j in range(npp):
                            add[pland, inew, j] += moved[p, i, j]
                D[d] = D[d] + add
        if pr is not None:                               # retours destination -> origine (coût croissant avec la durée)
            moved = D[d] * pr
            ret = moved.sum()
            if ret > 0:
                rcost = res.return_cost[t]
                D[d] = D[d] - moved
                add = np.zeros_like(D[o])
                for p in range(npos):
                    pland = int(res.land_d2o[p])         # atterrissage en rang (D7)
                    for i in range(nf):
                        inew = _snap(Wf, max(Wf[i] - rcost, 0.0))
                        for j in range(npp):
                            add[pland, inew, j] += moved[p, i, j]
                D[o] = D[o] + add
        departures[t], returns[t] = dep, ret
        # accumulation de patrimoine (snap) puis transition de position, par pays
        for c in (o, d):
            acc = np.zeros_like(D[c])
            for p in range(npos):
                for i in range(nf):
                    inew = nf_map[c][p, i]
                    for j in range(npp):
                        acc[p, inew, np_map[c][p, j]] += D[c][p, i, j]
            # transition de position (sur l'axe 0)
            D[c] = np.einsum("pq,pij->qij", P, acc)

    total_dep, total_ret = sum(departures.values()), sum(returns.values())
    peak = max(departures, key=lambda a: departures[a]) if total_dep > 0 else None
    return {"departures_by_age": departures, "peak_departure_age": peak,
            "emigrant_share": total_dep, "return_rate": (total_ret / total_dep) if total_dep > 0 else 0.0,
            "abroad_share_end": float(D[d].sum())}
