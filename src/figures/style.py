"""Socle de figures (research-repo-rules §4) : style, palette, sauvegarde bilingue, empreinte.

Les figures sont générées par code, en deux jeux EN et FR (paramètre de langue), déposées dans
``manuscript/{en,fr}/ssrn/figures/``, avec une empreinte de CONTENU calculée sur les données
tracées (pas sur les octets PNG, qui dépendent de la version du traceur). Aucun nombre tracé n'est
saisi à la main : chaque figure reçoit ses tableaux depuis le moteur.

Palette catégorielle unique, validée daltoniens (Okabe-Ito, script dataviz, CVD ΔE >= 8) ; le sens
de chaque couleur est fixé ici une fois pour toutes : une couleur = un pays, à toute occurrence.
Encodage secondaire systématique (marqueur par pays + étiquette directe) pour que l'identité ne
repose jamais sur la couleur seule.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.paths import ROOT as PROJECT_ROOT

# --- Palette : une couleur par pays, sens fixe (Okabe-Ito, CVD-safe) ---
COUNTRY_COLOR = {"fr": "#D55E00", "us": "#0072B2", "uk": "#E69F00", "ch": "#009E73",
                 "ca": "#CC79A7", "de": "#56B4E9", "it": "#999999", "es": "#F0E442", "jp": "#000000"}
COUNTRY_MARKER = {"fr": "o", "us": "s", "uk": "^", "ch": "D",
                  "ca": "v", "de": "P", "it": "X", "es": "*", "jp": "h"}
SEQ_HUE = "#0072B2"   # rampe séquentielle (magnitude), teinte unique clair->foncé
INK = "#222222"; MUTED = "#888888"; GRID = "#DDDDDD"

# --- Noms de pays par langue (axes/légendes dans la langue du manuscrit) ---
COUNTRY_NAME = {
    "fr": {"fr": "France", "us": "États-Unis", "uk": "Royaume-Uni", "ch": "Suisse",
           "ca": "Canada", "de": "Allemagne", "it": "Italie", "es": "Espagne", "jp": "Japon"},
    "en": {"fr": "France", "us": "United States", "uk": "United Kingdom", "ch": "Switzerland",
           "ca": "Canada", "de": "Germany", "it": "Italy", "es": "Spain", "jp": "Japan"},
}

# --- Étiquettes d'axes et termes, par langue ---
L = {
    "fr": {
        "percentile": "Percentile de gains", "theta": "Taux de conversion (%)",
        "cost_employer": "Coût employeur (milliers, monnaie locale)", "mass": "Masse de population",
        "age": "Âge", "income_index": "Indice de revenu d'âge", "share_payg": "Part répartition",
        "share_funded": "Part capitalisée", "ic": "Indice de coût de la vie (France = 100)",
        "pos": "Position de rang", "net_abroad": "Part nette à l'étranger", "amenity_cref": "Aménité (fraction de $c_{réf}$)",
        "kappa": "Coût de migration (milliers $ int.)", "mig_cost": "Coût de migration (milliers $ int.)",
        "dep_share": "Part de départs", "threshold_pos": "Position-seuil de départ",
        "with_housing": "avec propriété", "without_housing": "sans logement", "observed": "observé",
        "eta": "Appel de destination (fraction de $c_{réf}$)", "distance": "Distance de moments",
    },
    "en": {
        "percentile": "Earnings percentile", "theta": "Conversion rate (%)",
        "cost_employer": "Employer cost (thousands, local currency)", "mass": "Population mass",
        "age": "Age", "income_index": "Age-income index", "share_payg": "PAYG share",
        "share_funded": "Funded share", "ic": "Cost-of-living index (France = 100)",
        "pos": "Rank position", "net_abroad": "Net share abroad", "amenity_cref": "Amenity (fraction of $c_{ref}$)",
        "kappa": "Migration cost (thousands int. $)", "mig_cost": "Migration cost (thousands int. $)",
        "dep_share": "Departure share", "threshold_pos": "Departure threshold position",
        "with_housing": "with ownership", "without_housing": "without housing", "observed": "observed",
        "eta": "Destination appeal (fraction of $c_{ref}$)", "distance": "Moment distance",
    },
}

FIG_DIR = {lang: PROJECT_ROOT / "manuscript" / lang / "ssrn" / "figures" for lang in ("fr", "en")}


def setup():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def data_hash(*arrays) -> str:
    """Empreinte déterministe sur le CONTENU tracé (arrondi), pas sur les octets PNG."""
    h = hashlib.sha256()
    for a in arrays:
        try:
            flat = list(a.ravel())
        except AttributeError:
            flat = list(a) if hasattr(a, "__iter__") and not isinstance(a, str) else [a]
        for v in flat:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = 0.0 if (isinstance(v, float) and math.isnan(v)) else round(float(v), 6)
            h.update(repr(v).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def save(fig, name: str, lang: str, digest: str) -> str:
    FIG_DIR[lang].mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR[lang] / f"{name}.png", metadata={"Software": "stay-or-leave", "Creation Time": None})
    plt.close(fig)
    return digest
