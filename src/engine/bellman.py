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
    """Par position : (consommation réelle nette de loyer, épargne annuelle) en $ int."""
    pts = [p for p in load_cascade(country) if p.profile == "celibataire"]
    pts.sort(key=lambda p: p.cost_employer)
    ppp = col[country].ppp_aic_local_per_intl_usd
    sat = cal.consumption_saturation_intl_usd
    rent = (housing[country].rent_monthly_local * 12 / ppp) if country in housing else 0.0
    cons, save, positions, real = [], [], [], []
    for p in pts:
        real_pp = p.disposable_year / ppp
        after = max(0.0, real_pp - rent)
        c = min(after, sat)
        surplus = max(0.0, after - sat)
        contrib = pension[country].contrib_share_cost * p.cost_employer / ppp
        cons.append(c); save.append(surplus + contrib); positions.append(p.position); real.append(real_pp)
    return positions, np.array(cons), np.array(save), np.array(real)


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


def solve(origin: str, destination: str, col: Dict[str, CostOfLiving],
          pension: Dict[str, PensionParams], tax: Dict[str, CapitalTax],
          housing: Dict[str, Housing], cal: Calibration, dyn: Dynamics,
          nW: int = 25, Wmax: float = 6_000_000.0):
    """Résout la Bellman et renvoie, par âge, le seuil de départ (position/revenu réel)."""
    beta = 1.0 / (1.0 + cal.discount_rate_rho)
    r_eff = {c: (cal.r_mkt_certainty_equivalent if pension[c].regime == "funded" else cal.r_rep_real)
             for c in (origin, destination)}
    pos_o, cons_o, save_o, real_o = _country_flows(origin, col, pension, tax, housing, cal)
    pos_d, cons_d, save_d, real_d = _country_flows(destination, col, pension, tax, housing, cal)
    npos = len(pos_o)
    flows = {origin: (cons_o, save_o), destination: (cons_d, save_d)}
    Ptrans = _position_transition(npos)

    # Atterrissage EN RANG (percentile), pas par revenu réel : apparier les revenus réels
    # ré-injecterait le NIVEAU (la destination plus riche ferait "gagner" même en bas), le
    # double-compte interdit. Premier passage : phi=1, atterrissage au MÊME indice de percentile
    # (la valeur de destination à ce percentile porte déjà son niveau). L'effet de role_ratio
    # (translation de rang nette du niveau) est traité dans la couche statique D7 (seuil vs
    # sourcé), à injecter ici en décalage d'indice dans un raffinement ultérieur.
    identity = np.arange(npos)
    landing = {origin: identity, destination: identity}
    Wg = np.linspace(0.0, Wmax, nW)
    ages = list(range(dyn.age_start, dyn.age_retire + 1))
    countries = [origin, destination]

    def interpW(Vrow, Wq):
        return np.interp(Wq, Wg, Vrow)

    # valeur terminale : rente de retraite forfaitaire (4%) sur W, CRRA
    V = {c: np.zeros((npos, nW)) for c in countries}
    for c in countries:
        for ip in range(npos):
            V[c][ip] = _crra(0.04 * Wg + 1e-3, cal.risk_aversion)

    policy_leave, policy_return = {}, {}  # (age) -> (npos,nW) bool : partir / rentrer optimal
    for t in reversed(ages[:-1]):
        Vn = {c: np.zeros((npos, nW)) for c in countries}
        leave_grid = np.zeros((npos, nW), dtype=bool)              # partir optimal (origine -> dest)
        return_grid = np.zeros((npos, nW), dtype=bool)             # rentrer optimal (dest -> origine)
        for c in countries:
            cons_c, save_c = flows[c]
            A_here = dyn.amenity if c == origin else 0.0            # aménité = équivalent-conso en France
            for ip in range(npos):
                Wnext = Wg * (1 + r_eff[c]) + save_c[ip]            # accumulation du pot (sur la grille)
                EV_stay = sum(Ptrans[ip, jp] * interpW(V[c][jp], Wnext) for jp in range(npos))
                stay_val = _crra(cons_c[ip] + A_here, cal.risk_aversion) + beta * EV_stay  # array sur W
                best = stay_val
                if t <= dyn.age_mig_max:                            # option de migration
                    other = destination if c == origin else origin
                    cost = dyn.kappa if c == origin else dyn.lambda_return * dyn.kappa
                    cons_m, save_m = flows[other]
                    A_other = dyn.amenity if other == origin else 0.0
                    jp0 = int(landing[c][ip])                       # position d'atterrissage (rôle comparable)
                    Wm = np.maximum(Wg - cost, 0.0) * (1 + r_eff[other]) + save_m[jp0]
                    EV_move = sum(Ptrans[jp0, jp] * interpW(V[other][jp], Wm) for jp in range(npos))
                    move_val = _crra(cons_m[jp0] + A_other, cal.risk_aversion) + beta * EV_move
                    best = np.maximum(stay_val, move_val)
                    if c == origin:
                        leave_grid[ip] = move_val > stay_val
                    else:
                        return_grid[ip] = move_val > stay_val
                Vn[c][ip] = best
        V = Vn
        policy_leave[t] = leave_grid
        policy_return[t] = return_grid

    # W représentatif par âge : pot accumulé le long de la position médiane en restant en origine
    mid = npos // 2
    Wpath, w = {}, 0.0
    for t in ages:
        Wpath[t] = w
        w = w * (1 + r_eff[origin]) + save_o[mid]
    # seuils par âge au W représentatif : départ = plus basse position où partir est optimal ;
    # retour = plus haute position où rentrer est optimal (rentrer quand le revenu s'effondre).
    threshold, leave_idx, return_idx = {}, {}, {}
    for t in ages[:-1]:
        iw = int(np.argmin(np.abs(Wg - Wpath[t])))
        lrow = policy_leave[t][:, iw]
        rrow = policy_return[t][:, iw]
        li = next((i for i, v in enumerate(lrow) if v), None)
        ri = next((i for i in range(npos - 1, -1, -1) if rrow[i]), None)
        threshold[t] = (pos_o[li], real_o[li]) if li is not None else None
        leave_idx[t], return_idx[t] = li, ri
    return BellmanResult(ages=ages, positions=pos_o, real=real_o, threshold=threshold,
                         leave_idx=leave_idx, return_idx=return_idx, transition=Ptrans)


class BellmanResult:
    """Sortie du solveur : seuils de départ/retour par âge et transition de position."""

    def __init__(self, ages, positions, real, threshold, leave_idx, return_idx, transition):
        self.ages = ages
        self.positions = positions
        self.real = real
        self.threshold = threshold          # age -> (position, revenu réel) du seuil de départ
        self.leave_idx = leave_idx          # age -> plus basse position-index où partir (ou None)
        self.return_idx = return_idx        # age -> plus haute position-index où rentrer (ou None)
        self.transition = transition        # matrice de transition de position
