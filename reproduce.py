"""Point d'entrée déterministe de la reproduction (research-repo-rules §4).

État actuel : cascade France. Lit les composantes livrées, forme les deux taux de
conversion hors TVA (brut = salaire seul, élargi = prestations incluses), imprime une
empreinte SHA-256 pour le contrôle de déterminisme, et affiche la table. La taxe sur la
consommation n'intervient pas ici : elle vit dans la couche coût de la vie (IC_p, D3). Les
étapes aval (flux patrimonial, benchmark statique, Bellman, figures) se branchent au fur et
à mesure dans src/engine, src/analysis, src/figures.

Environnement : .venv-repro (voir requirements.txt). L'acquisition des paramètres a son
propre environnement et ne fait pas partie de cette chaîne.
"""
from __future__ import annotations

import hashlib

from src.common.paths import param_file
from src.data.france_cascade import load_cascade_fr
from src.engine.conversion import conversion_fr


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fr_path = param_file("fr", "theta_fr.csv")
    conv = conversion_fr(load_cascade_fr())

    print(f"[FR] cascade coût employeur -> revenu disponible (hors TVA) : "
          f"{len(conv)} points, sha256={_sha256(fr_path)[:16]}…")
    print(f"{'C (euros)':>12} {'theta_brut':>11} {'theta_elargi':>13}")
    for p in conv:
        print(f"{p.cost_employer_eur:>12} {p.theta_brut:>11.3f} {p.theta_elargi:>13.3f}")

    # Garde-fou économique : hors zone de transferts (coût employeur >= SMIC annuel),
    # θ brut décroît avec le revenu (régressivité) ; θ élargi >= θ brut partout (les
    # prestations ne peuvent qu'ajouter du revenu disponible).
    high = [p for p in conv if p.cost_employer_eur >= 28000]
    thetas = [p.theta_brut for p in high]
    assert thetas == sorted(thetas, reverse=True), "theta brut non décroissant au-dessus du SMIC"
    assert all(p.theta_elargi >= p.theta_brut - 1e-9 for p in conv), "theta élargi < theta brut"
    print("\nOK : theta brut décroissant au-dessus du SMIC ; theta élargi >= theta brut.")
    # TODO engine : flux patrimonial Pi, benchmark statique, Bellman, matrice de flux.


if __name__ == "__main__":
    main()
