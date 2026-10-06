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
