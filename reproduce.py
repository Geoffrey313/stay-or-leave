"""Point d'entrée déterministe de la reproduction (research-repo-rules §4).

État actuel : cascades socio-fiscales par pays (bloc D1). Pour chaque pays livré et chaque
profil familial (célibataire, parent isolé, couple deux enfants), lit les composantes
livrées, forme les deux taux de conversion hors taxe sur la consommation (brut = salaire
seul, élargi = prestations incluses), imprime une empreinte SHA-256 par pays pour le
contrôle de déterminisme, et affiche la table. La taxe sur la consommation vit dans la
couche coût de la vie (IC_p, D3). Les étapes aval (flux patrimonial, benchmark statique,
Bellman, figures) se branchent au fur et à mesure.

Environnement : .venv-repro (voir requirements.txt). L'acquisition des paramètres a ses
propres environnements et ne fait pas partie de cette chaîne.
"""
from __future__ import annotations

import hashlib
from itertools import groupby

from src.common.paths import param_file
from src.data.cascade import available_countries, load_cascade
from src.engine.conversion import conversion

SMIC_ANNUAL_COST = 28000  # borne de la zone de transferts (profil et pays de référence)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    countries = available_countries()
    print(f"[D1] cascades livrées : {', '.join(countries)}")
    for country in countries:
        conv = conversion(load_cascade(country))
        digest = _sha256(param_file(country, "cascade.csv"))[:16]
        print(f"\n=== {country.upper()} (hors taxe conso) sha256={digest}… ===")
        for profile, group in groupby(conv, key=lambda p: p.profile):
            pts = list(group)
            print(f"  profil : {profile}")
            print(f"  {'C (local)':>12} {'theta_brut':>11} {'theta_elargi':>13}")
            for p in pts:
                print(f"  {p.cost_employer:>12} {p.theta_brut:>11.3f} {p.theta_elargi:>13.3f}")
            high = [p.theta_brut for p in pts if p.cost_employer >= SMIC_ANNUAL_COST]
            assert high == sorted(high, reverse=True), f"theta brut non décroissant ({country}:{profile})"
            assert all(p.theta_elargi >= p.theta_brut - 1e-9 for p in pts), \
                f"theta élargi < brut ({country}:{profile})"

    print("\nOK : par pays et profil, theta brut décroissant au-dessus du seuil ; "
          "theta élargi >= theta brut.")
    # TODO engine : flux patrimonial Pi, benchmark statique, Bellman, matrice de flux.


if __name__ == "__main__":
    main()
