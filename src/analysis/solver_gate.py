"""Porte d'acceptation du solveur avant lecture de la SMM (critère d'arrêt objectif).

Le modèle dynamique a été corrigé par itérations (conversion rétroactive, régime phi=1, marginale
démographique). Pour ne pas rester dans un « encore un fix » sans fin, la SMM n'est lue QUE si le
solveur corrigé passe des conditions objectives, ancrées sur la reproduction, pas sur une
impression de propreté :

  1. la marginale transversale de rang = masses de population (Voronoi) à TOUT âge (test anti-dérive
     du M-H : la stationnaire de la transition est bien les masses, pas l'uniforme) ;
  2. l'aménité A est IDENTIFIABLE : le moment de retour doit devenir sensible à A (balayage). Test
     plus fin que « retour ~ 0,50 » : avant la re-spéc d'utilité, les moments étaient strictement
     invariants à A (0,273/0,019/0,931 pour A dans {45k,20k,5k}), donc A inestimable. On exige un
     écart non trivial du taux de retour sur le balayage de A ; sinon repli structurel déclaré ;
  3. choc permanent implicite de la transition = sigma_eta (D11) après M-H (la mobilité n'a pas
     dérivé en fixant la marginale) ;
  4. RÉTENTION immobilière (bloc D4) : le stock net à l'étranger est STRICTEMENT plus bas avec le
     canal propriété activé qu'avec (ownership ON < OFF), aux deux régimes de role_ratio. Mesurée
     sur la DISTRIBUTION (pas au point représentatif). Tient quel que soit le défaut du headline ;
     c'est une propriété du modèle (le logement retient), prouvée, pas affirmée.
  5. GRILLES de patrimoine NON MORDANTES (validité numérique, bloc C15) : les moments de tête sont
     invariants quand on double l'étendue des grilles W_f/W_p en préservant la résolution, et la
     masse terminale au plafond est négligeable. Sans cette condition, les chiffres du haut de
     distribution (verdict d'élite, SMM, encadrement) pourraient être des artefacts de troncature.

Les conditions 1-3 valident la lecture de la SMM (headline sans logement, qui atteint 6 %). La
condition 4 valide le résultat D4. La condition 5 valide que le solveur ne tronque pas le patrimoine
au sommet, là où vivent les résultats de tête. Une condition qui casse est un échec NOMMÉ.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def amenity_identifiability(col, pen, tax, house, cal, dyn,
                            a_values=(0, 5000, 10000)) -> dict:
    """Balaye A et renvoie le taux de retour simulé par A + l'écart. Re-spéc d'utilité réussie ssi
    le moment de retour varie non trivialement avec A (A inerte avant : écart ~ 0)."""
    from src.engine.bellman import solve
    from src.engine.simulate import simulate_moments
    by_a = {}
    for A in a_values:
        d = dyn.model_copy(update={"amenity": float(A)})
        by_a[A] = simulate_moments(solve("fr", "us", col, pen, tax, house, cal, d))["return_rate"]
    spread = max(by_a.values()) - min(by_a.values())
    return {"return_by_amenity": by_a, "spread": spread}


def retention_check(col, pen, tax, house, cal, dyn, a_ref=0.0, kappa_ref=20000.0,
                    role_ratios=(None, 1.27)) -> dict:
    """Condition 4 (bloc D4) : à params communs (migration positive), le stock net à l'étranger
    baisse-t-il avec le canal propriété activé ? Renvoie net ON/OFF par régime + le drapeau ON<OFF.
    Mesurée sur la DISTRIBUTION. Indépendante du défaut du headline."""
    from src.engine.bellman import solve
    from src.engine.simulate import simulate_moments
    d = dyn.model_copy(update={"amenity": float(a_ref), "kappa": float(kappa_ref)})
    by_regime, holds = {}, True
    for rr in role_ratios:
        off = simulate_moments(solve("fr", "us", col, pen, tax, house, cal, d, role_ratio=rr, ownership=False))
        on = simulate_moments(solve("fr", "us", col, pen, tax, house, cal, d, role_ratio=rr, ownership=True))
        by_regime[rr] = (off["abroad_share_end"], on["abroad_share_end"])
        holds = holds and (on["abroad_share_end"] < off["abroad_share_end"])
    return {"by_regime": by_regime, "holds": holds}


def grid_robustness_check(col, pen, tax, house, cal, dyn, role_ratios=(None, 1.27),
                          a_ref=0.0, kappa_ref=20000.0, tol=0.01) -> dict:
    """Condition 5 (VALIDITÉ, pas présentation) : les grilles de patrimoine W_f et W_p ne mordent
    pas. Si le plafond de grille sature au sommet (P99+, où vivent le verdict d'élite, la SMM et
    l'encadrement), les chiffres de tête sont des artefacts de troncature. Test d'INVARIANCE À
    L'ÉTENDUE : on double l'étendue des grilles ET le nombre de noeuds (résolution préservée) et on
    vérifie que les moments de tête (part d'émigrés, stock net à l'étranger) ne bougent pas, aux deux
    régimes de prime. Diagnostic complémentaire : masse terminale dans le bin de tête (snap au plus
    proche -> une grille qui mord empile la masse au plafond). `holds` = invariance sous tol."""
    from src.engine.bellman import solve
    from src.engine.simulate import simulate_moments
    d = dyn.model_copy(update={"amenity": float(a_ref), "kappa": float(kappa_ref)})
    by_regime, holds, worst = {}, True, 0.0
    for rr in role_ratios:
        base = simulate_moments(solve("fr", "us", col, pen, tax, house, cal, d, role_ratio=rr,
                                      nf=12, npp=8, Wf_max=6_000_000.0, Wp_max=2_000_000.0))
        wide = simulate_moments(solve("fr", "us", col, pen, tax, house, cal, d, role_ratio=rr,
                                      nf=24, npp=16, Wf_max=12_000_000.0, Wp_max=4_000_000.0))
        de = abs(wide["emigrant_share"] - base["emigrant_share"])
        da = abs(wide["abroad_share_end"] - base["abroad_share_end"])
        worst = max(worst, de, da)
        holds = holds and de <= tol and da <= tol
        by_regime[rr] = {"emig": (base["emigrant_share"], wide["emigrant_share"]),
                         "net": (base["abroad_share_end"], wide["abroad_share_end"]),
                         "top": (base["wf_top_mass"], base["wp_top_mass"])}
    return {"by_regime": by_regime, "worst_delta": worst, "tol": tol, "holds": holds}


def check_gate(res, sim: dict, sigma_eta_target: float, amenity_spread: float | None = None,
               retention: dict | None = None, grid: dict | None = None,
               emigrant_obs: float = 0.06, spread_min: float = 0.05) -> Dict[str, dict]:
    """Évalue les conditions. Renvoie {nom: {pass, value, detail}}. amenity_spread = écart du taux de
    retour sur le balayage de A (condition 2) ; retention = sortie de retention_check (condition 4).
    None si non évaluée."""
    pi = res.pop_mass
    P = res.transition

    # 1. anti-dérive : propager la marginale de la chaîne PURE depuis pi sur tout l'horizon
    m, worst = pi.copy(), 0.0
    for _ in range(len(res.ages)):
        worst = max(worst, float(np.max(np.abs(m - pi))))
        m = m @ P
    c1 = worst < 1e-4

    # 2. A identifiable : le taux de retour varie non trivialement avec A
    net = sim["abroad_share_end"]
    c2 = amenity_spread is not None and amenity_spread >= spread_min

    # 3. persistance : choc permanent implicite = sigma_eta (D11)
    c3 = abs(res.sigma_impl - sigma_eta_target) <= 5e-3

    sp = "n/a" if amenity_spread is None else f"{amenity_spread:.3f}"
    out = {
        "marginale_voronoi_tout_age": {"pass": c1, "value": worst,
                                       "detail": f"dérive max {worst:.2e} (seuil 1e-4)"},
        "amenite_identifiable": {"pass": c2, "value": amenity_spread,
                                 "detail": f"ecart taux retour sur balayage A = {sp} (seuil {spread_min}) ; net {net:.3f} vs {emigrant_obs} obs"},
        "persistance_sigma_eta": {"pass": c3, "value": res.sigma_impl,
                                  "detail": f"sigma implicite {res.sigma_impl:.4f} vs cible {sigma_eta_target}"},
    }
    # 4. rétention immobilière (D4) : net ON < OFF aux deux régimes, sur la distribution
    if retention is not None:
        reg = retention["by_regime"]
        det = " ; ".join(f"{'proxy' if rr is None else f'rr={rr}'}: {off:.3f}->{on:.3f}"
                         for rr, (off, on) in reg.items())
        out["retention_immobiliere"] = {"pass": bool(retention["holds"]), "value": None,
                                        "detail": f"net OFF->ON (doit baisser) : {det}"}
    # 5. grilles de patrimoine non mordantes (VALIDITÉ) : moments invariants quand on double
    #    l'étendue W_f/W_p (résolution préservée) ; masse terminale au plafond négligeable.
    if grid is not None:
        reg = grid["by_regime"]
        det = " ; ".join(
            f"{'proxy' if rr is None else f'rr={rr}'}: émig {v['emig'][0]:.3f}->{v['emig'][1]:.3f}, "
            f"net {v['net'][0]:.3f}->{v['net'][1]:.3f}, bin tête W_f/W_p {v['top'][0]:.1e}/{v['top'][1]:.1e}"
            for rr, v in reg.items())
        out["grille_non_mordante"] = {"pass": bool(grid["holds"]), "value": grid["worst_delta"],
                                      "detail": f"écart max en doublant l'étendue {grid['worst_delta']:.4f} "
                                                f"(seuil {grid['tol']}) ; {det}"}
    return out


def all_pass(gate: Dict[str, dict]) -> bool:
    return all(c["pass"] for c in gate.values())


def format_gate(gate: Dict[str, dict]) -> str:
    lines = []
    for name, c in gate.items():
        lines.append(f"    [{'PASS' if c['pass'] else 'FAIL'}] {name} : {c['detail']}")
    return "\n".join(lines)
