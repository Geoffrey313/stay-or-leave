"""Porte d'acceptation du solveur avant lecture de la SMM (critère d'arrêt objectif).

Le modèle dynamique a été corrigé par itérations (conversion rétroactive, régime phi=1, marginale
démographique). Pour ne pas rester dans un « encore un fix » sans fin, la SMM n'est lue QUE si le
solveur corrigé passe trois conditions objectives, ancrées sur la reproduction, pas sur une
impression de propreté :

  1. la marginale transversale de rang = masses de population (Voronoi) à TOUT âge (test anti-dérive
     du M-H : la stationnaire de la transition est bien les masses, pas l'uniforme) ;
  2. émigration dans l'ordre de grandeur observé SANS churn : taux de retour proche du 0,50 observé
     (pas 0,94), et stock net du bon ordre ;
  3. choc permanent implicite de la transition = sigma_eta (D11) après M-H (la mobilité n'a pas
     dérivé en fixant la marginale).

Si les trois passent, kappa*/A* se publient. Si l'une casse, c'est un échec NOMMÉ : on sait quoi
corriger, on ne devine pas.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def check_gate(res, sim: dict, sigma_eta_target: float,
               return_obs: float = 0.50, emigrant_obs: float = 0.06) -> Dict[str, dict]:
    """Évalue les trois conditions. Renvoie {nom: {pass, value, detail}}."""
    pi = res.pop_mass
    P = res.transition

    # 1. anti-dérive : propager la marginale de la chaîne PURE depuis pi sur tout l'horizon
    m, worst = pi.copy(), 0.0
    for _ in range(len(res.ages)):
        worst = max(worst, float(np.max(np.abs(m - pi))))
        m = m @ P
    c1 = worst < 1e-4

    # 2. pas de churn : taux de retour proche de l'observé, stock net du bon ordre
    ret, net = sim["return_rate"], sim["abroad_share_end"]
    c2 = abs(ret - return_obs) <= 0.15 and 0.2 * emigrant_obs <= net <= 3.0 * emigrant_obs

    # 3. persistance : choc permanent implicite = sigma_eta (D11)
    c3 = abs(res.sigma_impl - sigma_eta_target) <= 5e-3

    return {
        "marginale_voronoi_tout_age": {"pass": c1, "value": worst,
                                       "detail": f"dérive max {worst:.2e} (seuil 1e-4)"},
        "pas_de_churn": {"pass": c2, "value": ret,
                         "detail": f"taux retour {ret:.3f} vs {return_obs} obs ; net {net:.3f} vs {emigrant_obs} obs"},
        "persistance_sigma_eta": {"pass": c3, "value": res.sigma_impl,
                                  "detail": f"sigma implicite {res.sigma_impl:.4f} vs cible {sigma_eta_target}"},
    }


def all_pass(gate: Dict[str, dict]) -> bool:
    return all(c["pass"] for c in gate.values())


def format_gate(gate: Dict[str, dict]) -> str:
    lines = []
    for name, c in gate.items():
        lines.append(f"    [{'PASS' if c['pass'] else 'FAIL'}] {name} : {c['detail']}")
    return "\n".join(lines)
