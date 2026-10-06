"""Chemins portables, résolus relativement à la racine du dépôt.

Aucun chemin machine en dur (research-repo-rules §5). Tout le code passe par ici.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PARAMS = DATA / "params"


def param_file(*parts: str) -> Path:
    """Chemin d'un paramètre transformé livré, sous data/params/."""
    return PARAMS.joinpath(*parts)
