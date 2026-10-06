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
from src.data.housing import load_housing
from src.data.dynamics import load_dynamics
from src.engine.bellman import solve as bellman_solve
from src.engine.conversion import conversion
from src.engine.purchasing_power import real_purchasing_power
from src.engine.benchmark import stay_or_leave
from src.engine.pension import pension_switch
from src.engine.surplus import surplus_switch
from src.engine.patrimony import lifetime_patrimony, stay_or_leave_lifetime
from src.data.wage_premium import load_wage_premium, load_education_premium, load_occupation_premium
from src.engine.wage_premium import translate_rank, break_even_role_ratio

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
    print("    [HYPOTHÈSE phi=1 (même percentile) + flux annuel seul : PAS le résultat, cf. seuil D7]")
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
    house = load_housing()
    pat = {c: lifetime_patrimony(load_cascade(c), col, pen, tax, cal, house)
           for c in countries if c in pen}
    print("\n=== Valeur patrimoniale lifetime Pi : rester (FR) vs partir ===")
    print("    [HYPOTHÈSE phi=1 : PAS le résultat ; le résultat robuste est le seuil role_ratio* (D7)]")
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
    # --- D7 HEADLINE : seuil role_ratio* situé contre la prime sourcée (proxy) ---
    wp = load_wage_premium()
    print("\n=== D7 : SEUIL role_ratio* vs prime à rôle comparable SOURCÉE (proxy éducation) ===")
    print("    role_ratio* = prime réelle (dest/FR, net du niveau) à partir de laquelle partir domine.")
    print("    role_ratio sourcé = prime d'éducation tertiaire OCDE (proxy ; sous-estime les rôles")
    print("    d'élite US au sommet). Verdict : PARTIR si sourcé >= seuil. 'jamais' = dest ne rattrape pas.")
    for dest in ("us", "uk"):
        if dest not in pat:
            continue
        be = [b for b in break_even_role_ratio(pat["fr"], pat[dest]) if b.profile == "celibataire"]
        rr = wp[dest].role_ratio_ppp if dest in wp else None
        print(f"\n  France -> {dest.upper()}   (role_ratio sourcé = {rr:.2f})")
        print(f"  {'position':>8} {'role_ratio* seuil':>18} {'sourcé':>8}  verdict (proxy, pré-κ)")
        for b in be:
            star = b.role_ratio_star
            if star is None:
                verdict, s = "rester (jamais)", "jamais"
            else:
                verdict, s = ("PARTIR" if rr >= star else "rester"), f"{star:.2f}"
            print(f"  {b.position:>8} {s:>18} {rr:>8.2f}  {verdict}")
    # Sensibilité par niveau de diplôme (role_ratio = premium_dest / premium_FR)
    edu = load_education_premium()
    print("\n  Sensibilité role_ratio par niveau de diplôme (OCDE EaG) :")
    print(f"  {'niveau':>20} {'FR->US':>8} {'FR->UK':>8}")
    for att, lab in [("bachelor_6", "licence"), ("tertiary_5T8", "tertiaire"),
                     ("master_doctoral_7_8", "master/doctorat")]:
        us = edu["us"][att] / edu["fr"][att]
        uk = edu["uk"][att] / edu["fr"][att]
        print(f"  {lab:>20} {us:>8.2f} {uk:>8.2f}")
    print("    Constat : la France rémunère MIEUX les diplômes avancés (master/doctorat role_ratio")
    print("    US 0,98 < 1) ; l'avantage US est au niveau licence (1,06). Même au sommet ÉDUCATIF,")
    print("    rester domine. LIMITE : éducation != occupation ; les rôles d'élite US du sommet")
    print("    (tech/finance, P90-P99,9) exigent la rémunération occupationnelle (BLS OES / ONS ASHE).")
    print("    Friction (diplôme, langue, chômage) = kappa (D8), jamais phi.")

    # Test du sommet par OCCUPATION (role_ratio = prime_occ_US / prime_occ_FR)
    occ = load_occupation_premium()
    star_us = next((b.role_ratio_star for b in break_even_role_ratio(pat["fr"], pat["us"])
                    if b.profile == "celibataire" and b.position == "P90"), None)
    print(f"\n  TEST DU SOMMET par OCCUPATION (FR->US ; seuil role_ratio* ~1,03-1,14, P90={star_us}) :")
    print(f"  {'occupation (groupe)':>24} {'role_ratio':>11}  verdict au sommet")
    fr_mgr, fr_prof = occ["fr"]["managers"], occ["fr"]["professionals"]
    pairs = [("Management", "management", fr_mgr), ("Legal", "legal", fr_prof),
             ("Computer/math", "computer_mathematical", fr_prof),
             ("Healthcare", "healthcare_practitioners", fr_prof),
             ("Engineering", "architecture_engineering", fr_prof),
             ("Business/finance", "business_financial", fr_prof)]
    for lab, key, frp in pairs:
        rr = occ["us"][key] / frp
        verdict = "PARTIR" if (star_us and rr >= star_us) else "borderline/rester"
        print(f"  {lab:>24} {rr:>11.2f}  {verdict}")
    print("    => role_ratio OCCUPATION dépasse le seuil pour les rôles d'élite : au sommet")
    print("    (P90-P99,9, surreprésenté par ces métiers), PARTIR domine. Verdict OCCUPATION-DÉPENDANT :")
    print("    rester pour le diplômé moyen, partir pour les occupations d'élite. UK : groupes non")
    print("    sourçables proprement (médianes par groupe non accessibles). Friction = kappa (D8).")

    # --- Modèle dynamique : Bellman, seuil de départ par âge y*(t) ---
    dyn = load_dynamics()
    ages, bpos, breal, thr = bellman_solve("fr", "us", col, pen, tax, house, cal, dyn)
    print("\n=== Bellman dynamique : seuil de départ par âge, France -> US ===")
    print(f"    (état âge×pays×position×pot funded ; r_rep={cal.r_rep_real:.1%} vs funded EC="
          f"{cal.r_mkt_certainty_equivalent:.1%} ; kappa={dyn.kappa:.0f}, A={dyn.amenity:.0f}, lambda={dyn.lambda_return} ;")
    print(f"     migration bloquée > {dyn.age_mig_max} ans ; phi=1 atterrissage ; équivalent-certain)")
    print(f"  {'âge':>5}  seuil de départ (position | revenu réel $ int.)")
    for t in range(dyn.age_start, dyn.age_mig_max + 1, 5):
        v = thr.get(t)
        print(f"  {t:>5}  {'rester à toute position' if v is None else f'{v[0]} | {v[1]:.0f}'}")
    print("    Lecture : on part JEUNE (long horizon pour composer la bascule capitalisée), la")
    print("    migration s'éteignant avec l'âge (condition de régularité). phi=1 est optimiste :")
    print("    sous role_ratio occupationnel d'élite (>seuil) partir s'étend, sous le proxy éducation")
    print("    (<1) il se restreint. kappa/A posés [I] ; estimation sur flux migratoires = D13.")


if __name__ == "__main__":
    main()
