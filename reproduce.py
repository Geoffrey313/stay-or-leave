"""Point d'entrée déterministe de la reproduction (research-repo-rules §4).

État actuel : premier maillon exécutable. Lit le paramètre France livré (cascade
theta_FR(C)), le valide, en imprime une empreinte SHA-256 pour le contrôle de déterminisme,
et affiche la table. Les étapes aval (flux patrimonial, benchmark statique, Bellman,
figures) sont à brancher au fur et à mesure dans src/engine, src/analysis, src/figures.

Environnement : .venv-repro (voir requirements.txt). L'acquisition des paramètres a son
propre environnement et ne fait pas partie de cette chaîne.
"""
from __future__ import annotations

import hashlib

from src.common.paths import param_file
from src.data.france_cascade import load_theta_fr


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fr_path = param_file("fr", "theta_fr.csv")
    points = load_theta_fr()
    print(f"[FR] cascade theta_FR(C) : {len(points)} points, "
          f"sha256={_sha256(fr_path)[:16]}…")
    print(f"{'C (euros)':>12} {'net/an':>10} {'disp/an':>10} {'theta':>8}")
    for p in points:
        print(f"{p.cost_employer_eur:>12} {p.salaire_net_year_eur:>10} "
              f"{p.revenu_disponible_year_eur:>10} {p.theta_disposable:>8.3f}")

    # Garde-fou économique : hors zone de transferts, theta décroît avec le revenu
    # (régressivité). On le vérifie sur la partie haute (coût employeur >= SMIC annuel).
    high = [p for p in points if p.cost_employer_eur >= 28000]
    thetas = [p.theta_disposable for p in high]
    assert thetas == sorted(thetas, reverse=True), "theta non monotone décroissant au-dessus du SMIC"
    print("\nOK : theta décroissant au-dessus du SMIC (régressivité attendue).")
    # TODO engine : flux patrimonial Pi, benchmark statique, Bellman, matrice de flux.


if __name__ == "__main__":
    main()
