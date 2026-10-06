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
from src.data.cost_of_living import load_cost_of_living
from src.data.pension import load_pension, load_calibration
from src.data.capital_tax import load_capital_tax
from src.engine.conversion import conversion
from src.engine.purchasing_power import real_purchasing_power
from src.engine.benchmark import stay_or_leave
from src.engine.pension import pension_switch
from src.engine.surplus import surplus_switch
from src.engine.patrimony import lifetime_patrimony, stay_or_leave_lifetime

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
            print(f"  {'position':>8} {'C (local)':>12} {'theta_brut':>11} {'theta_elargi':>13}")
            for p in pts:
                print(f"  {p.position or '-':>8} {p.cost_employer:>12} "
                      f"{p.theta_brut:>11.3f} {p.theta_elargi:>13.3f}")
            high = [p.theta_brut for p in pts if p.cost_employer >= SMIC_ANNUAL_COST]
            assert high == sorted(high, reverse=True), f"theta brut non décroissant ({country}:{profile})"
            assert all(p.theta_elargi >= p.theta_brut - 1e-9 for p in pts), \
                f"theta élargi < brut ({country}:{profile})"

    print("\nOK : par pays et profil, theta brut décroissant au-dessus du seuil ; "
          "theta élargi >= theta brut.")

    # --- Premier benchmark statique : rester vs partir (pouvoir d'achat réel) ---
    col = load_cost_of_living()
    real = {c: real_purchasing_power(load_cascade(c), col) for c in countries}
    print("\n=== Benchmark statique : rester (FR) vs partir, pouvoir d'achat réel ($ int.) ===")
    print("    (profil célibataire, à position de distribution égale ; hors coût de migration)")
    for dest in ("us", "uk"):
        comps = [c for c in stay_or_leave(real["fr"], real[dest]) if c.profile == "celibataire"]
        print(f"\n  France -> {dest.upper()}")
        print(f"  {'position':>8} {'rester FR':>12} {'partir '+dest.upper():>12} {'gain partir':>12}  verdict")
        for c in comps:
            print(f"  {c.position:>8} {c.real_pp_origin:>12.0f} {c.real_pp_destination:>12.0f} "
                  f"{c.leave_gain_intl_usd:>12.0f}  {'PARTIR' if c.leave_dominates else 'rester'}")
    # --- Bascule retraite (D5) : patrimoine terminal gagné en quittant la répartition ---
    pen = load_pension()
    cal = load_calibration()
    if "fr" in pen:
        fr_ppp = col["fr"].ppp_aic_local_per_intl_usd
        fr_cascade = load_cascade("fr")
        print("\n=== Bascule retraite : gain de patrimoine terminal en quittant la France ===")
        print(f"    (r_rep={cal.r_rep_real:.1%} sans risque, r_mkt={cal.r_mkt_real:.1%} espérance risquée, "
              f"EC Merton={cal.r_mkt_certainty_equivalent:.1%} ; {cal.career_years} ans ; célibataire ; k$ int.)")
        print("    brut = borne haute AVANT risque (entre dans Pi avec sigma) ; "
              "EC = équivalent-certain illustratif (ajustement canonique = Bellman)")
        for dest in ("us", "uk"):
            if dest not in pen:
                continue
            sw = [s for s in pension_switch(fr_cascade, pen["fr"], pen[dest], cal)
                  if s.profile == "celibataire"]
            print(f"\n  France -> {dest.upper()} ({pen['fr'].regime} -> {pen[dest].regime})")
            print(f"  {'position':>8} {'cotis./an':>12} {'brut avant risque':>18} {'EC illustratif':>16}")
            for s in sw:
                print(f"  {s.position:>8} {s.annual_contribution:>12.0f} "
                      f"{s.switch_gain_gross_pre_risk / fr_ppp / 1000:>18.0f} "
                      f"{s.switch_gain_ce_illustrative / fr_ppp / 1000:>16.0f}")
    # --- Surplus d'épargne libre capitalisé net de fiscalité du capital (D6) ---
    tax = load_capital_tax()
    print("\n=== Surplus investi : patrimoine terminal gagné en quittant la France ===")
    print(f"    (au-dessus de {cal.consumption_saturation_intl_usd:.0f} $int. de saturation ; "
          f"net plus-values FR {tax['fr'].capgains_tax_rate:.1%} ; célibataire ; k$ int.)")
    print("    brut = avant risque (entre dans Pi avec sigma) ; EC = équivalent-certain illustratif")
    for dest in ("us", "uk"):
        if dest not in tax:
            continue
        sw = [s for s in surplus_switch(real["fr"], real[dest], tax, cal) if s.profile == "celibataire"]
        print(f"\n  France -> {dest.upper()} (plus-values {tax['fr'].capgains_tax_rate:.1%} -> "
              f"{tax[dest].capgains_tax_rate:.1%})")
        print(f"  {'position':>8} {'surplus FR':>11} {'surplus '+dest.upper():>11} "
              f"{'gain brut':>11} {'gain EC':>9}")
        for s in sw:
            print(f"  {s.position:>8} {s.surplus_origin:>11.0f} {s.surplus_destination:>11.0f} "
                  f"{s.wealth_gain_gross_pre_risk/1000:>11.0f} {s.wealth_gain_ce_illustrative/1000:>9.0f}")
    # --- Pi unique : valeur patrimoniale lifetime, rester vs partir (phi=1, baseline locataire) ---
    pat = {c: lifetime_patrimony(load_cascade(c), col, pen, tax, cal)
           for c in countries if c in pen}
    print("\n=== Valeur patrimoniale lifetime Pi : rester (FR) vs partir ===")
    print("    (phi=1 : position de destination EGALE, prime salariale NON intégrée ; baseline")
    print("     locataire ; équivalent-certain ; M$ int. en valeur présente ; célibataire)")
    for dest in ("us", "uk"):
        if dest not in pat:
            continue
        comps = [c for c in stay_or_leave_lifetime(pat["fr"], pat[dest]) if c.profile == "celibataire"]
        print(f"\n  France -> {dest.upper()}")
        print(f"  {'position':>8} {'Pi rester':>11} {'Pi partir':>11} {'gain partir':>12}  verdict")
        for c in comps:
            print(f"  {c.position:>8} {c.pi_origin_ce/1e6:>11.2f} {c.pi_destination_ce/1e6:>11.2f} "
                  f"{c.leave_gain_ce/1e6:>12.2f}  {'PARTIR' if c.leave_dominates_ce else 'rester'}")
    print("\n    NB : Pi à phi=1 n'est pas la réponse finale rester/partir ; D7 (prime salariale)")
    print("    relâche phi, D4 (immobilier) ajoute le logement ; la Bellman fera l'ajustement de")
    print("    risque canonique une fois (âge, réversibilité, coût de migration).")


if __name__ == "__main__":
    main()
