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
from src.engine.bellman import _country_flows as flows_for
from src.data.ownership import load_ownership
from src.engine.housing_channel import (monthly_payment as housing_monthly_payment,
                                        equity_schedule as housing_equity,
                                        net_sale_proceeds as housing_net_sale)
from src.engine.conversion import conversion
from src.engine.purchasing_power import real_purchasing_power
from src.engine.benchmark import stay_or_leave
from src.engine.pension import pension_switch
from src.engine.surplus import surplus_switch
from src.engine.patrimony import lifetime_patrimony, stay_or_leave_lifetime
from src.data.wage_premium import load_wage_premium, load_education_premium, load_occupation_premium
from src.engine.wage_premium import translate_rank, break_even_role_ratio
from src.engine.simulate import simulate_moments
from src.analysis.solver_gate import (check_gate, all_pass, format_gate,
                                       amenity_identifiability, retention_check)
from src.analysis.estimate_amenity_migration import estimate as estimate_amenity, estimate_amenity_curve

SMIC_ANNUAL_COST = 28000  # borne de la zone de transferts (profil et pays de référence)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(full: bool = False) -> None:
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

    # Test du sommet par OCCUPATION, SOURCÉ sur base homogène (SES 2018 FR/UK + OEWS 2023 US, médiane)
    occ = load_occupation_premium()
    star_us = next((b.role_ratio_star for b in break_even_role_ratio(pat["fr"], pat["us"])
                    if b.profile == "celibataire" and b.position == "P90"), None)
    fr_mgr, fr_prof = occ["fr"]["managers"], occ["fr"]["professionals"]
    print(f"\n  TEST DU SOMMET par OCCUPATION, sourcé base homogène (seuil role_ratio* ~1,03-1,14, P90={star_us}) :")
    print(f"  {'occupation (groupe)':>22} {'FR->US':>8} {'FR->UK':>8}  verdict au sommet (US)")
    rows = [("Dirigeants", occ["us"]["management"] / fr_mgr, occ["uk"]["managers"] / fr_mgr),
            ("Computer/math*", occ["us"]["computer_mathematical"] / fr_prof, occ["uk"]["professionals"] / fr_prof),
            ("Finance", occ["us"]["business_financial"] / fr_prof, occ["uk"]["professionals"] / fr_prof)]
    for lab, rr_us, rr_uk in rows:
        verdict = "PARTIR" if (star_us and rr_us >= star_us) else "borderline/rester"
        print(f"  {lab:>22} {rr_us:>8.2f} {rr_uk:>8.2f}  {verdict}")
    print("    (* Computer/math US = SOC-15 étroit vs ISCO-2 large -> role_ratio gonflé, à écarter.)")
    print("    => FR->US : role_ratio d'élite > 1 ROBUSTE (dirigeants 1,27, finance 1,22) ; au sommet")
    print("    (P90-P99,9, surreprésenté par ces métiers) PARTIR domine. FR->UK : role_ratio <= ~1")
    print("    (dirigeants 0,85 : la France sur-rémunère ses dirigeants vs le UK) -> l'élite US part,")
    print("    pas l'élite UK. Verdict OCCUPATION-DÉPENDANT et DESTINATION-dépendant. Friction = kappa (D8).")

    # --- Modèle dynamique : Bellman, seuil de départ par âge y*(t) ---
    dyn = load_dynamics()
    res = bellman_solve("fr", "us", col, pen, tax, house, cal, dyn)
    thr = res.threshold
    print("\n=== Bellman dynamique : seuil de départ par âge, France -> US ===")
    print(f"    (état âge×pays×position×pot funded W_f×droit répartition W_p ; r_rep={cal.r_rep_real:.1%} vs")
    print(f"     funded EC={cal.r_mkt_certainty_equivalent:.1%} ; kappa={dyn.kappa:.0f}, A={dyn.amenity:.0f}, "
          f"lambda={dyn.lambda_return}, delta={dyn.return_cost_slope} ;")
    print(f"     migration bloquée > {dyn.age_mig_max} ans ; atterrissage en RANG net du niveau role_ratio="
          f"{res.role_ratio:.2f} (D7, PLUS phi=1) ;")
    print(f"     utilité commensurable u(c/c_ref), c_ref={res.cref:.0f} ; terminal annuité par le même u)")
    print(f"  {'âge':>5}  seuil de départ (position | revenu réel $ int.)")
    for t in range(dyn.age_start, dyn.age_mig_max + 1, 5):
        v = thr.get(t)
        print(f"  {t:>5}  {'rester à toute position' if v is None else f'{v[0]} | {v[1]:.0f}'}")
    print("    Lecture : atterrissage en rang net du niveau (D7, role_ratio=1 redonne phi=1 ; proxy éducation")
    print("    0,97 ~ identité, queue d'occupation d'élite >1,1 étend le départ). Processus de revenu pondéré")
    print(f"    population : M-H vers masses Voronoi, p_move={res.p_move:.3f} -> sigma implicite={res.sigma_impl:.3f}")
    print("    = sigma_eta (D11). Utilité commensurable : conso en ratio c/c_ref, aménité A en équivalent-conso")
    print("    (estimande A/c_ref), terminal annuité (W_f au taux équivalent-certain net d'impôt, W_p au taux")
    print("    sûr) passé par le MÊME u -> agent averse au risque PARTOUT, l'ajustement D5/D6 mord sur W_f.")

    # --- Porte d'acceptation du solveur : quatre conditions (1-3 SMM headline OFF ; 4 rétention D4) ---
    sim = simulate_moments(res)
    sens = amenity_identifiability(col, pen, tax, house, cal, dyn)
    reten = retention_check(col, pen, tax, house, cal, dyn)
    gate = check_gate(res, sim, dyn.sigma_eta, amenity_spread=sens["spread"], retention=reten)
    print("\n=== PORTE DU SOLVEUR (critère d'arrêt avant SMM) ===")
    print(format_gate(gate))
    byA = " ".join(f"A={a:.0f}:retour={r:.3f}" for a, r in sens["return_by_amenity"].items())
    print(f"    balayage aménité -> {byA}")
    if all_pass(gate):
        print("    => les quatre conditions passent : SMM lue sur le headline sans logement (atteint 6%).")
        if full:
            # --- SMM COMPLÈTE (D13) : A et kappa en FONCTION du role_ratio (le seuil, pas le verdict) ---
            # Résultat de publication ; régénéré par `python reproduce.py --full` (déterministe, lourd).
            curve = estimate_amenity_curve()
            print("\n=== SMM COMPLÈTE (D13) : A et kappa en fonction de la prime d'élite role_ratio (SOURCÉE) ===")
            print("    Le signe de A dépend de la prime d'occupation d'élite, DÉSORMAIS SOURCÉE (Eurostat SES")
            print("    2018 FR + BLS OEWS 2023 US, cf. occupation_premium.csv). On publie A/kappa en FONCTION")
            print("    du role_ratio le long de la borne sourcée (discipline D7 : le seuil, pas le verdict).")
            print(f"  {'role_ratio':>10} {'A*/c_ref':>9} {'A* ($)':>8} {'kappa* ($)':>11} {'dist':>6}  statut / source")
            for p in curve:
                tag = " [A EN BORNE BASSE]" if p["A_at_lower_bound"] else (" [A en borne haute]" if p["A_at_upper_bound"] else "")
                print(f"  {p['role_ratio']:>10.2f} {p['A']/res.cref:>9.2f} {p['A']:>8.0f} {p['kappa']:>11.0f}"
                      f" {p['distance']:>6.3f}  {p['kind']}{tag}")
            print("    Lecture : flag levé, mais HIÉRARCHIE de robustesse. Estimation CENTRALE = A au")
            print("    PLANCHER (0 au proxy 0,97 et à la moyenne d'élite 1,09) : la formule capée restaure le")
            print("    plancher (le plafond OASDI amplifie l'avantage capitalisé US au sommet, via le surplus,")
            print("    ce qui compense). A>0 (~0,32 c_ref, médiane 1,27) est le BORD OPTIMISTE, PAS une")
            print("    moitié co-égale : il dépend")
            print("    du choix de la médiane ET d'un pont de source unique US/FR (Eurostat SES 2018 ISCO ->")
            print("    BLS OEWS 2023 SOC, inter-source/année/classif, qui joue EN FAVEUR). Le UK est le cell")
            print("    PROPRE (FR et UK même source/année/classif) : role_ratio <= ~1 (dirigeants 0,85) ->")
            print("    A au plancher, résultat SOLIDE. Énoncé : l'aménité est nulle/proche de zéro ; un")
            print("    attachement positif n'apparaît qu'à la marge, vers les US, sur les médianes et via un")
            print("    pont favorable ; absent vers le UK. Résidu : tech/finance en proxy ISCO-2 (SES keyless).")
        else:
            # --- Point SOURCÉ (défaut rapide) : SMM au role_ratio du proxy éducation ---
            est = estimate_amenity()
            s = est["moments_sim"]
            bound = " [A EN BORNE BASSE]" if est["A_at_lower_bound"] else ""
            print("\n=== SMM, point sourcé (défaut) : A et kappa au role_ratio proxy éducation ===")
            print(f"    A* = {est['best_A']:.0f} (A/c_ref = {est['best_A']/res.cref:.2f}){bound} , "
                  f"kappa* = {est['best_kappa']:.0f}  (distance {est['distance']:.3f})")
            print(f"    moments : pic={s['peak_departure_age']}, net étranger={s['abroad_share_end']:.3f},"
                  f" retour={s['return_rate']:.3f} ; observés {est['moments_obs']}")
            print("    A central au PLANCHER sous la formule retraite capée (plafond OASDI US -> l'élite")
            print("    s'appuie sur le surplus r_mkt, avantage capitalisé amplifié au sommet). Le bord")
            print("    optimiste A>0 vit à l'élite US-médiane : python reproduce.py --full")
        print("    Résidus déclarés : pic de départ et taux de retour non parfaits (moments grossiers D13).")
    else:
        print("    => une condition CASSE : estimation NON publiée (échec nommé, pas un soupçon).")

    # --- D4 canal propriété : faisabilité (accession, équité, rétention) ; sourcé, pré-intégration ---
    own, ref_size = load_ownership()
    print("\n=== D4 : canal propriété, lecture de faisabilité (logement réf. "
          f"{ref_size:.0f} m2) ===")
    print("    Accession déterministe au seuil y_immo (DTI) ; principal remboursé = épargne forcée")
    print("    (équité W_h, actif terminal) ; vente EXONÉRÉE à frais à la migration = rétention.")
    print(f"  {'pays':>5} {'y_immo $int':>11} {'accèdent':>26} {'équité term.':>12} {'rétention (frais)':>17}")
    for c in ("fr", "us", "uk"):
        o = own[c]; ppp = col[c].ppp_aic_local_per_intl_usd
        p_real = o.reference_price(ref_size) / ppp
        principal = p_real * (1 - o.down_payment_share); down = p_real * o.down_payment_share
        y_immo = housing_monthly_payment(principal, o.mortgage_rate, o.term_years) * 12 / o.dti_max
        pos, _, _, _, real = flows_for(c, col, pen, tax, house, cal)
        acced = ",".join(p for p, r in zip(pos, real) if r >= y_immo)
        eq_term = housing_equity(principal, o.mortgage_rate, o.term_years, down)[-1]
        retention = eq_term - housing_net_sale(eq_term, o.sale_fee_share)
        print(f"  {c:>5} {y_immo:>11.0f} {acced:>26} {eq_term:>12.0f} {retention:>17.0f}")
    print("    Lecture : en France, accession dès P50 ; posséder est MOINS cher en flux que louer")
    print("    (intérêts < loyer de marché) et bâtit une équité terminale ~206k$int -> renforce rester")
    print("    au milieu/haut. Vente à la migration = frais de transaction (FR 5%, US 7%, UK 2%) sur")
    print("    l'équité : +30 à +50% de coût de sortie effectif pour un propriétaire. Loyer imputé")
    print("    DIFFÉRENTIEL (jamais ajout absolu, déjà dans IC_p).")

    # --- D4 : ENCADREMENT de l'émigration (borne basse sans logement / borne haute lock-in) ---
    print("\n=== D4 : encadrement par le canal propriété (switch documenté, OFF = headline) ===")
    dref = dyn.model_copy(update={"amenity": 0.0, "kappa": 20000.0})
    print("    Stock net à l'étranger à params communs (A=0, kappa=20000), propriété OFF vs ON :")
    for rr, lab in [(None, "proxy éducation"), (1.27, "élite US médiane (sourcée)")]:
        off = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, dref, role_ratio=rr, ownership=False))
        on = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, dref, role_ratio=rr, ownership=True))
        print(f"      {lab:>22} : OFF={off['abroad_share_end']:.3f} (borne basse) -> ON={on['abroad_share_end']:.3f} (borne haute)")
    print("    Deux RÉSULTATS, pas un gap à décomposer :")
    print("    (i) la rétention par le logement est de PREMIER ORDRE (condition 4, signée aux deux régimes) ;")
    print("    (ii) imposé UNIVERSELLEMENT et déterministe, le lock-in SUR-RETIENT : net plafonne ~4% même")
    print("         à migration quasi gratuite, sous les 6% observés -> régime RÉFUTÉ par les flux, au même")
    print("         titre que phi=1 (A*=0 y est un ARTEFACT de sur-rétention, pas une absence d'aménité).")
    print("    Le modèle ENCADRE donc l'émigration : borne basse sans logement (atteint 6% mais ignore la")
    print("    force logement prouvée), borne haute à lock-in universel (sur-retient à ~4%) ; le 6% observé")
    print("    est entre les deux. La fourchette est IDENTIFIÉE. La marge de mobilité non modélisée (~2 pts)")
    print("    est compatible avec plusieurs mécanismes que ce modèle NE SÉPARE PAS : (1) louer plutôt que")
    print("    posséder (tenure endogène) ; (2) GARDER-ET-LOUER plutôt que vendre (évite la vente forcée,")
    print("    d'où vient la sur-rétention) ; (3) dénominateur MEAE « Français » != cadres (écart de")
    print("    population) ; (4) prime d'élite à la marge (borne sourcée 1,07-1,27 straddle le seuil A>0,")
    print("    médiane vs moyenne ; tech/finance en proxy ISCO-2). La localiser dans la")
    print("    fourchette relève de la tenure endogène et/ou du garder-et-louer : travaux futurs.")
    print("    Propriété-ON reste un SWITCH (ownership=True), jamais le défaut tant que ces marges ne sont")
    print("    pas modélisées. Solveur 3 accumulateurs = baseline sans logement bit-à-bit quand OFF.")


if __name__ == "__main__":
    import sys
    main(full="--full" in sys.argv)
