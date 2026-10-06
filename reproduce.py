"""Point d'entrée déterministe de la reproduction (research-repo-rules §4).

État actuel : cascade France pour trois profils familiaux (célibataire, parent isolé,
couple deux enfants). Lit les composantes livrées, forme les deux taux de conversion hors
TVA (brut = salaire seul, élargi = prestations incluses), imprime une empreinte SHA-256
pour le contrôle de déterminisme, et affiche la table par profil. La taxe sur la
consommation n'intervient pas ici : elle vit dans la couche coût de la vie (IC_p, D3). Les
étapes aval (flux patrimonial, benchmark statique, Bellman, figures) se branchent au fur et
à mesure dans src/engine, src/analysis, src/figures.

Environnement : .venv-repro (voir requirements.txt). L'acquisition des paramètres a son
propre environnement et ne fait pas partie de cette chaîne.
"""
from __future__ import annotations

import hashlib
from itertools import groupby

from src.common.paths import param_file
from src.data.france_cascade import load_cascade_fr
from src.engine.conversion import conversion_fr

SMIC_ANNUAL_COST = 28000  # borne de la zone de transferts (coût employeur annuel au SMIC)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fr_path = param_file("fr", "theta_fr.csv")
    conv = conversion_fr(load_cascade_fr())
    print(f"[FR] cascade coût employeur -> revenu disponible (hors TVA), 3 profils : "
          f"{len(conv)} points, sha256={_sha256(fr_path)[:16]}…")

    for profile, group in groupby(conv, key=lambda p: p.profile):
        pts = list(group)
        print(f"\n  profil : {profile}")
        print(f"  {'C (euros)':>12} {'theta_brut':>11} {'theta_elargi':>13}")
        for p in pts:
            print(f"  {p.cost_employer_eur:>12} {p.theta_brut:>11.3f} {p.theta_elargi:>13.3f}")
        # Garde-fou : hors zone de transferts, θ brut décroît (régressivité) ; θ élargi >= θ brut.
        high = [p.theta_brut for p in pts if p.cost_employer_eur >= SMIC_ANNUAL_COST]
        assert high == sorted(high, reverse=True), f"theta brut non décroissant ({profile})"
        assert all(p.theta_elargi >= p.theta_brut - 1e-9 for p in pts), f"theta élargi < brut ({profile})"

    print("\nOK : par profil, theta brut décroissant au-dessus du SMIC ; theta élargi >= theta brut.")
    # TODO engine : flux patrimonial Pi, benchmark statique, Bellman, matrice de flux.


if __name__ == "__main__":
    main()
