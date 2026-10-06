"""Simulation de cohorte sur la politique de la Bellman -> moments de flux migratoires.

Propagation DÉTERMINISTE d'une distribution de masse sur les positions de revenu (pas de
Monte Carlo) : à chaque âge, on applique la politique (partir si position >= seuil de départ ;
rentrer si position <= seuil de retour), puis la transition de position. On en tire les moments
que la méthode des moments simulés (D13) confronte aux flux observés : profil du taux de départ
par âge (et son pic), part cumulée d'émigrés, taux de retour.
"""
from __future__ import annotations

import numpy as np


def _stationary(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def simulate_moments(res, age_mig_max: int) -> dict:
    """Moments simulés à partir d'un BellmanResult."""
    P = res.transition
    npos = len(res.positions)
    fr = _stationary(P).copy()          # distribution de positions à l'entrée (toute en France)
    us = np.zeros(npos)
    ages = res.ages
    departures, returns, fr_before = {}, {}, {}
    for t in ages[:-1]:
        fr_before[t] = fr.sum()
        li, ri = res.leave_idx.get(t), res.return_idx.get(t)
        dep = ret = 0.0
        if li is not None and t <= age_mig_max:          # départs (haut de distribution)
            moved = fr.copy(); moved[:li] = 0.0
            dep = moved.sum(); fr = fr - moved; us = us + moved
        if ri is not None:                                # retours (bas de distribution, option)
            moved = us.copy(); moved[ri + 1:] = 0.0
            ret = moved.sum(); us = us - moved; fr = fr + moved
        departures[t], returns[t] = dep, ret
        fr = P.T @ fr                                     # transition de position
        us = P.T @ us
    total_dep = sum(departures.values())
    total_ret = sum(returns.values())
    peak_age = max(departures, key=lambda a: departures[a]) if total_dep > 0 else None
    return {
        "departures_by_age": departures,
        "peak_departure_age": peak_age,
        "emigrant_share": total_dep,                      # part cumulée ayant émigré (brut)
        "return_rate": (total_ret / total_dep) if total_dep > 0 else 0.0,
        "abroad_share_end": float(us.sum()),
    }
