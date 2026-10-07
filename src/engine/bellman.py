"""Modèle dynamique : équation de Bellman du cycle de vie (récursion arrière).

État = (âge t, pays courant p, position de distribution du revenu, pot funded W_f, droit
répartition W_p). La position évolue dans le temps (transition type marche aléatoire sur les
percentiles livrés) — c'est la composante permanente du revenu. Le flux instantané
(consommation réelle nette de loyer) et l'épargne annuelle (cotisation retraite + surplus) à
chaque (pays, position) viennent DIRECTEMENT du moteur statique, sans pont approximatif.

Deux accumulateurs (correctness) : le patrimoine EXISTANT ne bascule pas de régime avec la
résidence. W_f (actifs) croît à r_mkt équivalent-certain NET de l'impôt sur plus-values de la
RÉSIDENCE et suit l'agent ; W_p (droit répartition, vesté) croît à r_rep PARTOUT. La résidence
ne change que le flux, la fiscalité du capital sur W_f, et le ROUTAGE des nouvelles cotisations
(répartition -> W_p ; capitalisation -> W_f).

Atterrissage en DÉCALAGE DE RANG net du niveau (bloc D7, PLUS phi=1) : chaque distribution porte
déjà son niveau, donc role_ratio n'est PAS un match de revenu réel inter-pays (réinjection du
niveau) mais un multiplicateur de rémunération DANS la distribution d'arrivée. L'agent au rang ip,
occupation rémunérée role_ratio fois (vs l'origine, net du niveau), atterrit au rang de
destination payant role_ratio*(réel d'arrivée au rang ip) ; symétrique, 1/role_ratio, en rentrant.
role_ratio=1 redonne l'ancien phi=1 « même rang ». C'est le NIVEAU de l'incitation migratoire : il
doit être le bon avant de lire kappa/A.

Coût de retour CROISSANT avec la durée à l'étranger (bloc D8) : lambda*kappa*(1+delta*frac),
frac = fraction de carrière écoulée (proxy de la durée passée au loin — on perd réseau et
reconnaissance du diplôme en restant parti). frac se lit sur l'âge car la migration est un
événement quasi unique avant age_mig_max ; delta est posé [I] et testé en sensibilité. Ce coût
porte l'identifiant du MOMENT de retour (donc de A) : à lambda*kappa plat, le retour était trop
bon marché et le taux de retour sur-prédit.

Actions : rester, partir (coût kappa, âge <= age_mig_max), rentrer (coût de retour ci-dessus).
Aménité A perçue en France. Utilité CRRA sur la consommation et sur la rente terminale (4% sur
W_f + W_p). Récursion arrière exacte (horizon fini). grid-agnostique.

Simplifications DÉCLARÉES : transition de position tridiagonale calibrée (pas estimée) ; rente
terminale forfaitaire ; role_ratio statique D7 (coupe, borne haute du premium causal) ; frac par
l'âge (proxy de durée, exact pour la cohorte dominante qui part une fois autour de 30 ans).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.data.cascade import load_cascade
from src.data.capital_tax import CapitalTax
from src.data.cost_of_living import CostOfLiving
from src.data.housing import Housing
from src.data.pension import Calibration, PensionParams
from src.data.dynamics import Dynamics
from src.data.wage_premium import load_wage_premium


def _crra(c: np.ndarray, gamma: float) -> np.ndarray:
    c = np.maximum(c, 1e-6)
    return np.log(c) if abs(gamma - 1.0) < 1e-9 else (c ** (1 - gamma) - 1) / (1 - gamma)


def _country_flows(country: str, col, pension, tax, housing, cal):
    """Par position : consommation, épargne FUNDED (surplus + cotisation si capitalisation),
    cotisation PAYG (si répartition), revenu réel. En $ int. Le routage de la cotisation retraite
    dépend du régime du pays (capitalisation -> funded r_mkt ; répartition -> droit PAYG r_rep)."""
    pts = [p for p in load_cascade(country) if p.profile == "celibataire"]
    pts.sort(key=lambda p: p.cost_employer)
    ppp = col[country].ppp_aic_local_per_intl_usd
    sat = cal.consumption_saturation_intl_usd
    rent = (housing[country].rent_monthly_local * 12 / ppp) if country in housing else 0.0
    funded = pension[country].regime == "funded"
    cons, save_f, save_p, positions, real = [], [], [], [], []
    for p in pts:
        real_pp = p.disposable_year / ppp
        after = max(0.0, real_pp - rent)
        surplus = max(0.0, after - sat)
        contrib = pension[country].contrib_share_cost * p.cost_employer / ppp
        cons.append(min(after, sat))
        save_f.append(surplus + (contrib if funded else 0.0))   # -> pot funded (r_mkt)
        save_p.append(0.0 if funded else contrib)               # -> droit répartition (r_rep)
        positions.append(p.position); real.append(real_pp)
    return positions, np.array(cons), np.array(save_f), np.array(save_p), np.array(real)


def _position_transition(n: int, p_move: float = 0.4) -> np.ndarray:
    """Transition tridiagonale sur les positions (persistance + mobilité symétrique)."""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1 - p_move
        if i > 0: P[i, i - 1] += p_move / 2
        else: P[i, i] += p_move / 2
        if i < n - 1: P[i, i + 1] += p_move / 2
        else: P[i, i] += p_move / 2
    return P


def _landing_maps(real_o: np.ndarray, real_d: np.ndarray, role_ratio: float):
    """Atterrissage en DÉCALAGE DE RANG net du niveau (D7). L'ancrage de chaque distribution porte
    déjà l'écart de niveau : role_ratio n'est donc PAS un match de revenu réel d'un pays à l'autre
    (ce serait réinjecter le niveau), mais un multiplicateur de rémunération DANS la distribution
    d'arrivée. Partir : l'agent au rang ip, dont l'occupation est rémunérée role_ratio fois (vs
    l'origine, net du niveau), atterrit au rang de DESTINATION payant role_ratio*(réel de dest au
    rang ip). Rentrer : symétrique, 1/role_ratio dans la distribution d'ORIGINE. role_ratio=1 donne
    l'identité (phi=1, même rang) ; role_ratio>1 décale vers le haut, <1 vers le bas. Les deux
    cartes s'ancrent sur le pays d'ARRIVÉE (jamais d'inter-niveau). Renvoie (o2d, d2o)."""
    o2d = np.array([int(np.argmin(np.abs(real_d - real_d[ip] * role_ratio))) for ip in range(len(real_o))])
    d2o = np.array([int(np.argmin(np.abs(real_o - real_o[ip] / role_ratio))) for ip in range(len(real_d))])
    return o2d, d2o


def _interp2(V2d, wf_g, wp_g, wf_new, wp_new):
    """Interpolation bilinéaire séparable de V2d(Wf,Wp) aux nouveaux noeuds (wf_new x wp_new)."""
    tmp = np.empty((len(wf_g), len(wp_g)))
    for j in range(len(wp_g)):
        tmp[:, j] = np.interp(wf_new, wf_g, V2d[:, j])     # le long de Wf
    out = np.empty_like(tmp)
    for i in range(len(wf_g)):
        out[i, :] = np.interp(wp_new, wp_g, tmp[i, :])     # le long de Wp
    return out


def solve(origin: str, destination: str, col: Dict[str, CostOfLiving],
          pension: Dict[str, PensionParams], tax: Dict[str, CapitalTax],
          housing: Dict[str, Housing], cal: Calibration, dyn: Dynamics,
          role_ratio: float | None = None,
          nf: int = 12, npp: int = 8, Wf_max: float = 6_000_000.0, Wp_max: float = 2_000_000.0):
    """Résout la Bellman à deux accumulateurs avec atterrissage en rang et coût de retour croissant.

    role_ratio : prime à rôle comparable (destination/origine, réel, net du niveau, bloc D7). Si
    None, lu dans wage_premium.csv pour la destination (1.0 à défaut = ancien phi=1). Le patrimoine
    existant ne change pas de régime avec la résidence ; seuls le flux, l'impôt du capital sur W_f
    et le routage des nouvelles cotisations changent. Rente terminale forfaitaire (4%) sur W_f+W_p.
    """
    if role_ratio is None:
        wp = load_wage_premium()
        role_ratio = wp[destination].role_ratio_ppp if destination in wp else 1.0
    beta = 1.0 / (1.0 + cal.discount_rate_rho)
    rce = cal.r_mkt_certainty_equivalent
    r_f = {c: rce * (1 - tax[c].capgains_tax_rate) for c in (origin, destination)}  # funded net impôt résidence
    r_p = cal.r_rep_real                                                            # droit répartition, partout
    pos_o, cons_o, sf_o, sp_o, real_o = _country_flows(origin, col, pension, tax, housing, cal)
    pos_d, cons_d, sf_d, sp_d, real_d = _country_flows(destination, col, pension, tax, housing, cal)
    npos = len(pos_o)
    flows = {origin: (cons_o, sf_o, sp_o), destination: (cons_d, sf_d, sp_d)}
    o2d, d2o = _landing_maps(real_o, real_d, role_ratio)       # atterrissage en rang (D7)
    land = {origin: o2d, destination: d2o}                      # pays courant -> indices d'atterrissage
    Ptrans = _position_transition(npos)
    Wf_g = np.linspace(0.0, Wf_max, nf)
    Wp_g = np.linspace(0.0, Wp_max, npp)
    ages = list(range(dyn.age_start, dyn.age_retire + 1))
    countries = [origin, destination]
    span = max(1, dyn.age_retire - dyn.age_start)
    # coût de retour croissant avec la durée à l'étranger (proxy par l'âge) ; départ = kappa plat
    return_cost = {t: dyn.lambda_return * dyn.kappa * (1 + dyn.return_cost_slope * (t - dyn.age_start) / span)
                   for t in ages}

    # valeur terminale : CRRA d'une rente forfaitaire 4% sur le patrimoine total (W_f + W_p)
    WF, WP = np.meshgrid(Wf_g, Wp_g, indexing="ij")
    Vterm = _crra(0.04 * (WF + WP) + 1e-3, cal.risk_aversion)
    V = {c: [Vterm.copy() for _ in range(npos)] for c in countries}

    policy_leave, policy_return = {}, {}
    for t in reversed(ages[:-1]):
        Vn = {c: [None] * npos for c in countries}
        leave_grid = np.zeros((npos, nf, npp), dtype=bool)
        return_grid = np.zeros((npos, nf, npp), dtype=bool)
        for c in countries:
            cons_c, sf_c, sp_c = flows[c]
            A_here = dyn.amenity if c == origin else 0.0
            for ip in range(npos):
                wf_n = Wf_g * (1 + r_f[c]) + sf_c[ip]            # accumulation funded (sur grille)
                wp_n = Wp_g * (1 + r_p) + sp_c[ip]               # accumulation droit répartition
                EV_stay = sum(Ptrans[ip, jp] * _interp2(V[c][jp], Wf_g, Wp_g, wf_n, wp_n)
                              for jp in range(npos))
                stay_val = _crra(cons_c[ip] + A_here, cal.risk_aversion) + beta * EV_stay
                best = stay_val
                if t <= dyn.age_mig_max:
                    other = destination if c == origin else origin
                    cost = dyn.kappa if c == origin else return_cost[t]       # retour croissant
                    il = int(land[c][ip])                                     # atterrissage en rang (D7)
                    cons_m, sf_m, sp_m = flows[other]
                    A_other = dyn.amenity if other == origin else 0.0
                    wf_m = np.maximum(Wf_g - cost, 0.0) * (1 + r_f[other]) + sf_m[il]  # κ payé sur W_f
                    wp_m = Wp_g * (1 + r_p) + sp_m[il]                                 # W_p vesté, inchangé
                    EV_move = sum(Ptrans[il, jp] * _interp2(V[other][jp], Wf_g, Wp_g, wf_m, wp_m)
                                  for jp in range(npos))
                    move_val = _crra(cons_m[il] + A_other, cal.risk_aversion) + beta * EV_move
                    best = np.maximum(stay_val, move_val)
                    if c == origin:
                        leave_grid[ip] = move_val > stay_val
                    else:
                        return_grid[ip] = move_val > stay_val
                Vn[c][ip] = best
        V = Vn
        policy_leave[t] = leave_grid
        policy_return[t] = return_grid

    # (W_f, W_p) représentatifs par âge : accumulés le long de la position médiane en restant
    mid = npos // 2
    Wpath, wf, wp2 = {}, 0.0, 0.0
    for t in ages:
        Wpath[t] = (wf, wp2)
        wf = wf * (1 + r_f[origin]) + sf_o[mid]
        wp2 = wp2 * (1 + r_p) + sp_o[mid]
    threshold, leave_idx, return_idx = {}, {}, {}
    for t in ages[:-1]:
        iwf = int(np.argmin(np.abs(Wf_g - Wpath[t][0])))
        iwp = int(np.argmin(np.abs(Wp_g - Wpath[t][1])))
        lcol = policy_leave[t][:, iwf, iwp]
        rcol = policy_return[t][:, iwf, iwp]
        li = next((i for i, v in enumerate(lcol) if v), None)
        ri = next((i for i in range(npos - 1, -1, -1) if rcol[i]), None)
        threshold[t] = (pos_o[li], real_o[li]) if li is not None else None
        leave_idx[t], return_idx[t] = li, ri
    return BellmanResult(ages=ages, positions=pos_o, real=real_o, threshold=threshold,
                         leave_idx=leave_idx, return_idx=return_idx, transition=Ptrans,
                         policy_leave=policy_leave, policy_return=policy_return,
                         Wf_g=Wf_g, Wp_g=Wp_g, r_f=r_f, r_p=r_p, flows=flows,
                         origin=origin, destination=destination, age_mig_max=dyn.age_mig_max,
                         kappa=dyn.kappa, return_cost=return_cost, role_ratio=role_ratio,
                         land_o2d=o2d, land_d2o=d2o)


class BellmanResult:
    """Sortie du solveur : seuils de départ/retour par âge, transition et cartes d'atterrissage."""

    def __init__(self, ages, positions, real, threshold, leave_idx, return_idx, transition,
                 policy_leave=None, policy_return=None, Wf_g=None, Wp_g=None, r_f=None, r_p=None,
                 flows=None, origin=None, destination=None, age_mig_max=None,
                 kappa=None, return_cost=None, role_ratio=None, land_o2d=None, land_d2o=None):
        self.ages = ages
        self.positions = positions
        self.real = real
        self.threshold = threshold          # age -> (position, revenu réel) du seuil de départ
        self.leave_idx = leave_idx          # age -> plus basse position-index où partir (ou None)
        self.return_idx = return_idx        # age -> plus haute position-index où rentrer (ou None)
        self.transition = transition        # matrice de transition de position
        # Pour la simulation de cohorte sur la distribution jointe (position, W_f, W_p) :
        self.policy_leave = policy_leave    # age -> (npos, nf, npp) bool
        self.policy_return = policy_return
        self.Wf_g, self.Wp_g = Wf_g, Wp_g
        self.r_f, self.r_p = r_f, r_p       # rendements funded (par pays) et répartition
        self.flows = flows                  # pays -> (cons, save_funded, save_payg) par position
        self.origin, self.destination = origin, destination
        self.age_mig_max = age_mig_max
        self.kappa = kappa                  # coût de départ (plat)
        self.return_cost = return_cost      # age -> coût de retour (croissant avec la durée)
        self.role_ratio = role_ratio        # prime à rôle comparable (D7) utilisée à l'atterrissage
        self.land_o2d = land_o2d            # indices d'atterrissage origine -> destination
        self.land_d2o = land_d2o            # indices d'atterrissage destination -> origine
