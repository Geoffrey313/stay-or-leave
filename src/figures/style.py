"""Socle d'export de données de figures (research-repo-rules §4). Les figures sont dessinées EN
LaTeX (pgfplots pour les courbes, TikZ pour les schémas) ; ce module n'exporte que les DONNÉES,
en fichiers ``.dat`` déposés dans ``manuscript/{en,fr}/ssrn/figures/`` (identiques des deux côtés,
les libellés d'axes vivant dans le .tex, dans la langue du manuscrit). Empreinte de contenu sur les
données écrites, pour le contrôle de déterminisme. Aucun nombre tracé n'est saisi à la main.

La palette et les styles de trait sont fixés UNE fois dans le préambule LaTeX (ColorBrewer sobre :
France bleu, États-Unis bordeaux, Royaume-Uni orange, Suisse vert ; + style de trait par pays pour
la lisibilité en noir et blanc). Ils ne sont pas répétés ici.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from src.common.paths import ROOT as PROJECT_ROOT

FIG_DIR = {lang: PROJECT_ROOT / "manuscript" / lang / "ssrn" / "figures" for lang in ("fr", "en")}


def _fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{round(float(v), 6):g}"


def write_dat(name: str, header: list[str], rows) -> str:
    """Écrit `rows` (itérable de tuples) en table d'espaces avec `header`, dans les deux dossiers de
    figures (données identiques). Renvoie une empreinte de contenu."""
    lines = [" ".join(header)]
    h = hashlib.sha256()
    for r in rows:
        cells = [_fmt(v) for v in r]
        lines.append(" ".join(cells))
        h.update(" ".join(cells).encode()); h.update(b"\n")
    text = "\n".join(lines) + "\n"
    for lang in ("fr", "en"):
        FIG_DIR[lang].mkdir(parents=True, exist_ok=True)
        (FIG_DIR[lang] / f"{name}.dat").write_text(text)
    return h.hexdigest()[:16]
