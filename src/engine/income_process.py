"""Processus de revenu de cycle de vie (bloc D11) pour la Bellman.

log y_t = g(t) + z_t, avec g(t) profil déterministe en cloche (croissance jusqu'au pic vers
50 ans puis reflux, cadre) et z_t marche aléatoire (z_t = z_{t-1} + eta_t, eta ~ N(0, sigma^2)),
la composante permanente étant la variable d'état stochastique. z est discrétisé sur une grille
bornée avec une matrice de transition gaussienne (style Tauchen pour une marche aléatoire).
Déclaré : g(t) est une calibration de forme (cadre), pas une estimation.
"""
from __future__ import annotations

import numpy as np


def age_profile(ages: np.ndarray, peak_age: float = 50.0, peak_gain: float = 0.42) -> np.ndarray:
    """g(t) en niveau (facteur multiplicatif), normalisé à 1 à 40 ans ; cloche, pic ~50.

    Cadre : coût employeur médian ~+42% du début de carrière au pic 50-54 puis reflux.
    """
    a = ages - 40.0
    # parabole concave calibrée : sommet ~ peak_age, redescente douce ensuite
    b = peak_gain / ((peak_age - 25.0) ** 2)
    g = 1.0 + 2 * b * (peak_age - 40.0) * a - b * a ** 2
    return np.maximum(g, 0.3)


def zgrid(n: int = 11, zmax: float = 2.0):
    """Grille de la composante permanente (log), symétrique."""
    return np.linspace(-zmax, zmax, n)


def transition_matrix(zs: np.ndarray, sigma_eta: float) -> np.ndarray:
    """Matrice de transition de la marche aléatoire : P(z'|z) ~ densité N(z, sigma^2) sur la grille."""
    n = len(zs)
    P = np.zeros((n, n))
    for i, zi in enumerate(zs):
        dens = np.exp(-0.5 * ((zs - zi) / sigma_eta) ** 2)
        P[i] = dens / dens.sum()
    return P


# --- Transition sur les ANCRES DE PERCENTILE (grille inégalement espacée) -------------------
# Les ancres P10..P99,9 échantillonnent la queue bien plus densément que le corps. Une marche
# tridiagonale à probabilités égales sur l'indice a une stationnaire UNIFORME, qui sur-pondère
# la queue (3/8 sur le top-3 contre ~7,5% de masse réelle). La stationnaire correcte sur les
# noeuds est la LARGEUR DE RANG que chaque ancre sous-tend (masse Voronoi) : c'est elle qui rend
# la distribution transversale de rang uniforme. On retargette la tridiagonale vers ces masses
# par Metropolis-Hastings à bilan détaillé, et on calibre la mobilité pour que le choc permanent
# implicite colle à sigma_eta (D11).

def percentile_of(label: str) -> float:
    """'P99.9' -> 99.9 ; 'P10' -> 10."""
    return float(label[1:])


def voronoi_masses(labels) -> np.ndarray:
    """Masse de population par ancre = largeur de rang Voronoi (mi-chemins entre ancres voisines,
    bornes 0 et 100), normalisée. C'est la stationnaire cible (transversale de rang uniforme)."""
    p = np.array([percentile_of(l) for l in labels], dtype=float)
    edges = np.concatenate([[0.0], (p[1:] + p[:-1]) / 2, [100.0]])
    m = np.diff(edges)
    return m / m.sum()


def tridiagonal_proposal(n: int, p_move: float) -> np.ndarray:
    """Proposition symétrique : persistance 1-p_move, mobilité locale p_move/2 vers chaque voisin
    (réfléchie aux bords). Symétrique, donc M-H se réduit à l'acceptation min(1, pi_j/pi_i)."""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1 - p_move
        if i > 0: P[i, i - 1] += p_move / 2
        else: P[i, i] += p_move / 2
        if i < n - 1: P[i, i + 1] += p_move / 2
        else: P[i, i] += p_move / 2
    return P


def mh_transition(Q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings à bilan détaillé : retargette la proposition Q vers la stationnaire pi.
    pi_i P_ij = pi_j P_ji par construction, donc power-iteration(P) = pi exactement. La masse
    rejetée retombe sur la diagonale (persistance). Préserve le caractère local de Q."""
    n = len(pi)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or Q[i, j] <= 0:
                continue
            accept = min(1.0, (pi[j] * Q[j, i]) / (pi[i] * Q[i, j]))
            P[i, j] = Q[i, j] * accept
        P[i, i] = 1.0 - P[i, :].sum()
    return P


def implied_sigma_eta(P: np.ndarray, log_vals: np.ndarray, pi: np.ndarray) -> float:
    """Choc permanent implicite = écart-type conditionnel de Delta(log-revenu) sur un pas, moyenné
    par la stationnaire. Pour la marche aléatoire z'=z+eta, cet écart-type cible sigma_eta (D11)."""
    cond_var = np.array([np.sum(P[i, :] * (log_vals - log_vals[i]) ** 2) for i in range(len(pi))])
    return float(np.sqrt(np.sum(pi * cond_var)))


def calibrate_position_transition(labels, real_vals, sigma_eta: float):
    """Transition sur les ancres : M-H(tridiagonale, pi=Voronoi) avec la mobilité p_move calibrée
    pour que le choc permanent implicite = sigma_eta (D11). Renvoie (P, pi, p_move, sigma_impliqué).
    M-H fixe la marginale pour TOUT p_move, donc p_move règle librement la persistance."""
    pi = voronoi_masses(labels)
    logv = np.log(np.asarray(real_vals, dtype=float))
    n = len(pi)

    def sigma_at(pm: float) -> float:
        return implied_sigma_eta(mh_transition(tridiagonal_proposal(n, pm), pi), logv, pi)

    lo, hi = 1e-4, 0.98                         # sigma croît avec p_move -> bissection monotone
    if sigma_at(hi) < sigma_eta:                # cible inatteignable : mobilité maximale
        pm = hi
    elif sigma_at(lo) > sigma_eta:              # déjà trop mobile : mobilité minimale
        pm = lo
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sigma_at(mid) > sigma_eta: hi = mid
            else: lo = mid
        pm = 0.5 * (lo + hi)
    P = mh_transition(tridiagonal_proposal(n, pm), pi)
    return P, pi, pm, implied_sigma_eta(P, logv, pi)
