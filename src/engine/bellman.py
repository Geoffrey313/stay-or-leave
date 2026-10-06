"""Modèle dynamique : équation de Bellman du cycle de vie (récursion arrière).

État = (âge t, pays courant p, position de distribution du revenu, patrimoine funded W).
La position évolue dans le temps (transition type marche aléatoire sur les percentiles livrés)
— c'est la composante permanente du revenu. Le flux instantané (consommation réelle nette de
loyer) et l'épargne annuelle (cotisation retraite + surplus) à chaque (pays, position) viennent
DIRECTEMENT du moteur statique, sans pont approximatif. Le pot funded W croît au rendement du
régime du pays courant : répartition -> r_rep (quasi sans risque) ; capitalisation -> taux
ÉQUIVALENT-CERTAIN r_mkt - 0,5*gamma*sigma^2 (l'ajustement de risque CRRA, fait UNE fois, porté
par le taux). Actions : rester, partir (coût kappa, âge <= age_mig_max), rentrer (coût
lambda*kappa). Aménité A perçue en France. Utilité CRRA sur la consommation et sur la rente de
retraite terminale tirée de W. Récursion arrière exacte (horizon fini).

Simplifications DÉCLARÉES (premier passage) : phi=1 pour l'atterrissage (même percentile ;
role_ratio = sensibilité statique D7, à injecter en décalage de rang plus tard) ; transition de
position tridiagonale calibrée (pas estimée) ; rente terminale forfaitaire sur W. grid-agnostique.
À RAFFINER (limite connue) : W croît au rendement du régime du pays COURANT sur TOUT le pot, ce
qui fait basculer rétroactivement les cotisations passées à r_mkt en partant et SUR-INCITE la
migration précoce (seuil trop bas). Correctif à venir : séparer le pot FUNDED (r_mkt, suit l'agent)
du droit RÉPARTITION (r_rep, acquis dans le pays où cotisé), comme le fait le moteur statique.
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
          nf: int = 12, npp: int = 8, Wf_max: float = 6_000_000.0, Wp_max: float = 2_000_000.0):
    """Résout la Bellman à deux accumulateurs (funded W_f, droit répartition W_p).

    Correctness : le patrimoine EXISTANT ne bascule pas de régime avec la résidence. W_f (actifs)
    croît à r_mkt équivalent-certain NET de l'impôt sur les plus-values de la RÉSIDENCE ; W_p
    (droit répartition, vesté) croît à r_rep PARTOUT. La résidence ne change que le flux, la
    fiscalité du capital sur W_f, et le ROUTAGE des nouvelles cotisations retraite (répartition
    -> W_p ; capitalisation -> W_f). Atterrissage en rang (phi=1, même percentile). Rente
    terminale forfaitaire (4%) sur W_f + W_p.
    """
    beta = 1.0 / (1.0 + cal.discount_rate_rho)
    rce = cal.r_mkt_certainty_equivalent
    r_f = {c: rce * (1 - tax[c].capgains_tax_rate) for c in (origin, destination)}  # funded net impôt résidence
    r_p = cal.r_rep_real                                                            # droit répartition, partout
    pos_o, cons_o, sf_o, sp_o, real_o = _country_flows(origin, col, pension, tax, housing, cal)
    pos_d, cons_d, sf_d, sp_d, real_d = _country_flows(destination, col, pension, tax, housing, cal)
    npos = len(pos_o)
    flows = {origin: (cons_o, sf_o, sp_o), destination: (cons_d, sf_d, sp_d)}
    Ptrans = _position_transition(npos)
    Wf_g = np.linspace(0.0, Wf_max, nf)
    Wp_g = np.linspace(0.0, Wp_max, npp)
    ages = list(range(dyn.age_start, dyn.age_retire + 1))
    countries = [origin, destination]

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
                    cost = dyn.kappa if c == origin else dyn.lambda_return * dyn.kappa
                    cons_m, sf_m, sp_m = flows[other]
                    A_other = dyn.amenity if other == origin else 0.0
                    wf_m = np.maximum(Wf_g - cost, 0.0) * (1 + r_f[other]) + sf_m[ip]  # κ payé sur W_f
                    wp_m = Wp_g * (1 + r_p) + sp_m[ip]                                 # W_p vesté, inchangé
                    EV_move = sum(Ptrans[ip, jp] * _interp2(V[other][jp], Wf_g, Wp_g, wf_m, wp_m)
                                  for jp in range(npos))
                    move_val = _crra(cons_m[ip] + A_other, cal.risk_aversion) + beta * EV_move
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
    Wpath, wf, wp = {}, 0.0, 0.0
    for t in ages:
        Wpath[t] = (wf, wp)
        wf = wf * (1 + r_f[origin]) + sf_o[mid]
        wp = wp * (1 + r_p) + sp_o[mid]
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
                         kappa=dyn.kappa, lambda_kappa=dyn.lambda_return * dyn.kappa)


class BellmanResult:
    """Sortie du solveur : seuils de départ/retour par âge et transition de position."""

    def __init__(self, ages, positions, real, threshold, leave_idx, return_idx, transition,
                 policy_leave=None, policy_return=None, Wf_g=None, Wp_g=None, r_f=None, r_p=None,
                 flows=None, origin=None, destination=None, age_mig_max=None,
                 kappa=None, lambda_kappa=None):
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
        self.kappa, self.lambda_kappa = kappa, lambda_kappa
