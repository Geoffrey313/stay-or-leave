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
from src.engine.income_process import calibrate_position_transition


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
    pp = pension[country]
    cons, save_f, save_p, positions, real = [], [], [], [], []
    for p in pts:
        real_pp = p.disposable_year / ppp
        after = max(0.0, real_pp - rent)
        surplus = max(0.0, after - sat)
        contrib_payg, contrib_funded = pp.split(p.cost_employer, ppp)  # formule capée per-position
        cons.append(min(after, sat))
        save_f.append(surplus + contrib_funded)
        save_p.append(contrib_payg)
        positions.append(p.position); real.append(real_pp)
    return positions, np.array(cons), np.array(save_f), np.array(save_p), np.array(real)


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


def _batch_interp(slices, Wf_g, wfa, Wp_g, wp):
    """Interp bilinéaire séparable en LOT sur l'axe de durée d (bloc D4). slices (nd,nf,npp) ;
    wfa (nd,nf) points sur W_f ; wp (npp,) points sur W_p. Renvoie (nd,nf,npp). d n'intervient QUE
    par l'empilement (axe 0, gather d'indice pur) : on interpole W_f puis W_p, jamais d. Même
    sémantique que _interp2 (linéaire, extrapolation constante aux bords, comme np.interp)."""
    nd, nf, npp = slices.shape
    # formule et bornage IDENTIQUES à np.interp (vérifié bit-à-bit) : slope*(x-x0)+lo, dx clampé
    idx = np.clip(np.searchsorted(Wf_g, wfa), 1, nf - 1)                 # (nd,nf)
    x0 = Wf_g[idx - 1]; x1 = Wf_g[idx]
    dx = np.clip(wfa - x0, 0.0, x1 - x0)[:, :, None]                     # (nd,nf,1)
    idx_e = idx[:, :, None].repeat(npp, axis=2)
    lo = np.take_along_axis(slices, idx_e - 1, axis=1)
    hi = np.take_along_axis(slices, idx_e, axis=1)
    val = (hi - lo) / (x1 - x0)[:, :, None] * dx + lo                    # f(wfa, Wp_g)
    j = np.clip(np.searchsorted(Wp_g, wp), 1, npp - 1)                   # (npp,)
    y0 = Wp_g[j - 1]; y1 = Wp_g[j]
    dy = np.clip(wp - y0, 0.0, y1 - y0)                                  # (npp,)
    lo2 = val[:, :, j - 1]; hi2 = val[:, :, j]
    return (hi2 - lo2) / (y1 - y0) * dy + lo2


def _country_primitives(c, col, pension):
    """Par pays : positions, revenu réel, cotisation retraite SCINDÉE (répartition -> W_p ;
    capitalisée -> W_f), PPP. Sans loyer (le logement est traité par tenure dans le solveur, D4)."""
    pts = [p for p in load_cascade(c) if p.profile == "celibataire"]
    pts.sort(key=lambda p: p.cost_employer)
    ppp = col[c].ppp_aic_local_per_intl_usd
    pp = pension[c]
    real = np.array([p.disposable_year / ppp for p in pts])
    splits = [pp.split(p.cost_employer, ppp) for p in pts]            # formule capée per-position
    contrib_payg = np.array([s[0] for s in splits])
    contrib_funded = np.array([s[1] for s in splits])
    pos = [p.position for p in pts]
    return pos, real, contrib_payg, contrib_funded, ppp


def _housing_params(c, own, ref, ppp, real, npos, ownership):
    """Paramètres propriété réels (bloc D4) : mensualité M, seuil d'accession thr (rang), équité par
    durée, apport, frais. ownership=False -> thr=npos (personne ne possède, baseline locataire)."""
    from src.engine.housing_channel import monthly_payment, equity_schedule
    o = own[c]
    P = o.reference_price(ref) / ppp
    principal = P * (1 - o.down_payment_share); down = P * o.down_payment_share
    M = monthly_payment(principal, o.mortgage_rate, o.term_years) * 12
    y_immo = M / o.dti_max
    thr = (next((i for i, r in enumerate(real) if r >= y_immo), npos) if ownership else npos)
    eq = equity_schedule(principal, o.mortgage_rate, o.term_years, down)   # d=0..term
    return {"M": M, "thr": thr, "term": o.term_years, "eq": eq, "down": down,
            "fee": o.sale_fee_share, "y_immo": y_immo}


def _tenure_flow(ip, H, real_val, contrib_payg, contrib_funded, rent_val, sat, nd, dvec):
    """Flux par durée d selon tenure (bloc D4). Propriétaire (rang>=thr) : sort la mensualité M tant
    que d<=terme puis 0 (prêt éteint = loyer imputé émergent) ; le principal part en équité (capté
    par d, pas ici). Locataire : sort le loyer de marché. La cotisation retraite est SCINDÉE :
    part capitalisée -> épargne funded (sf), part répartition -> droit W_p (sp).
    Renvoie cons(nd), sf(nd), sp(scalaire)."""
    owner = ip >= H["thr"]
    hout = np.where(dvec <= H["term"], H["M"], 0.0) if owner else np.full(nd, rent_val)
    after = real_val - hout
    cons = np.minimum(after, sat)
    surplus = np.maximum(0.0, after - sat)
    sf = surplus + contrib_funded     # -> pot funded (r_mkt)
    sp = contrib_payg                 # -> droit répartition (r_rep)
    return cons, sf, sp


def _transition_maps(ip_from, wf_n, H, netsaleA, nd, Dcap, npos, down):
    """Pour chaque ip2 : (dn (nd,), wfa (nd,nf)) de la transition de tenure, d en index PUR.
    owner->owner : d+1 ; renter->owner : achat (apport sur W_f, d=1) ; owner->renter : vente
    intra-pays (produit net sur W_f, d=0) ; renter->renter : d=0."""
    owner_now = ip_from >= H["thr"]
    dvecp1 = np.clip(np.arange(nd) + 1, 0, Dcap)
    dn_map, wfa_map = {}, {}
    for ip2 in range(npos):
        owner_next = ip2 >= H["thr"]
        if owner_next:
            if owner_now:
                dn = dvecp1; wfa = wf_n
            else:
                dn = np.ones(nd, int); wfa = np.maximum(wf_n - down, 0.0)
        else:
            dn = np.zeros(nd, int)
            wfa = wf_n + (netsaleA[:, None] if owner_now else 0.0)
        dn_map[ip2] = dn; wfa_map[ip2] = wfa
    return dn_map, wfa_map


def solve(origin: str, destination: str, col: Dict[str, CostOfLiving],
          pension: Dict[str, PensionParams], tax: Dict[str, CapitalTax],
          housing: Dict[str, Housing], cal: Calibration, dyn: Dynamics,
          role_ratio: float | None = None, ownership: bool = False,
          nf: int = 12, npp: int = 8, Wf_max: float = 6_000_000.0, Wp_max: float = 2_000_000.0):
    """Résout la Bellman à TROIS accumulateurs (W_f funded, W_p répartition, W_h équité logement).

    W_h n'est PAS une grille continue : l'équité est déterministe à durée de détention donnée, donc
    un axe ENTIER d = années de détention dans le pays courant (pur décalage d'indice, zéro
    interpolation sur d ; l'interp en lot ne touche que W_f et W_p). Tenure déterministe par rang
    (propriétaire ssi rang >= seuil d'accession y_immo). L'apport sort de W_f à l'achat ; la vente
    (changement de rang ou MIGRATION) rend l'équité nette de frais à W_f, EXONÉRÉE de plus-value
    (résidence principale). La vente forcée à la migration est le mécanisme de RÉTENTION (bloc D4).
    ownership=False : personne ne possède (baseline locataire), reproduit le solveur sans logement.

    Deux accumulateurs financiers inchangés : W_f (actifs) au taux équivalent-certain net d'impôt de
    résidence, W_p (répartition, vesté) au taux sûr. Atterrissage en rang net du niveau (role_ratio,
    D7). Coût de retour croissant avec la durée (D8). Utilité commensurable u(c/c_ref), terminal
    annuité par le même u (aversion partout). role_ratio=1 redonne phi=1.
    """
    if role_ratio is None:
        wp = load_wage_premium()
        role_ratio = wp[destination].role_ratio_ppp if destination in wp else 1.0
    from src.data.ownership import load_ownership
    own, ref = load_ownership()
    beta = 1.0 / (1.0 + cal.discount_rate_rho)
    g = cal.risk_aversion
    rce = cal.r_mkt_certainty_equivalent
    r_f = {c: rce * (1 - tax[c].capgains_tax_rate) for c in (origin, destination)}
    r_p = cal.r_rep_real
    sat = cal.consumption_saturation_intl_usd
    floor = cal.subsistence_floor_intl_usd
    prim = {c: _country_primitives(c, col, pension) for c in (origin, destination)}
    pos_o, real_o, contrib_payg_o, contrib_funded_o, ppp_o = prim[origin]
    npos = len(pos_o)
    med = pos_o.index("P50") if "P50" in pos_o else npos // 2
    cref = float(real_o[med])
    rent = {c: (housing[c].rent_monthly_local * 12 / prim[c][4]) if c in housing else 0.0
            for c in (origin, destination)}
    real_c = {origin: real_o, destination: prim[destination][1]}
    contrib_payg_c = {origin: contrib_payg_o, destination: prim[destination][2]}
    contrib_funded_c = {origin: contrib_funded_o, destination: prim[destination][3]}
    H = {c: _housing_params(c, own, ref, prim[c][4], prim[c][1], npos, ownership)
         for c in (origin, destination)}
    Dcap = max(H[origin]["term"], H[destination]["term"]) if ownership else 0
    nd = Dcap + 1
    dvec = np.arange(nd)

    def eqarr(c):
        e = H[c]["eq"]
        return np.array([e[d] if d < len(e) else e[-1] for d in range(nd)])
    eqA = {c: eqarr(c) for c in (origin, destination)}
    netsaleA = {c: np.maximum(eqA[c], 0.0) * (1 - H[c]["fee"]) for c in (origin, destination)}

    o2d, d2o = _landing_maps(real_o, prim[destination][1], role_ratio)
    land = {origin: o2d, destination: d2o}
    Ptrans, pop_mass, p_move, sigma_impl = calibrate_position_transition(pos_o, real_o, dyn.sigma_eta)
    Wf_g = np.linspace(0.0, Wf_max, nf)
    Wp_g = np.linspace(0.0, Wp_max, npp)
    ages = list(range(dyn.age_start, dyn.age_retire + 1))
    countries = [origin, destination]
    span = max(1, dyn.age_retire - dyn.age_start)
    return_cost = {t: dyn.lambda_return * dyn.kappa * (1 + dyn.return_cost_slope * (t - dyn.age_start) / span)
                   for t in ages}
    T_ret = cal.retirement_years
    ann_util = sum(beta ** k for k in range(T_ret))
    annf = lambda r: (r / (1 - (1 + r) ** (-T_ret))) if abs(r) > 1e-9 else 1.0 / T_ret

    # valeur terminale par (pays, rang, durée) : patrimoine annuité par le MÊME u ; W_f au taux EC
    # net d'impôt, W_p au taux sûr, équité W_h(d) réalisée au taux sûr ; aménité à domicile.
    WF, WP = np.meshgrid(Wf_g, Wp_g, indexing="ij")
    V = {c: np.empty((npos, nd, nf, npp)) for c in countries}
    for c in countries:
        for ip in range(npos):
            owner = ip >= H[c]["thr"]
            for d in range(nd):
                E = eqA[c][d] if owner else 0.0
                rf = WF * annf(r_f[c]) + WP * annf(r_p) + E * annf(r_p) + (dyn.amenity if c == origin else 0.0)
                V[c][ip, d] = ann_util * _crra(np.maximum(rf, floor) / cref, g)

    def cont(c, ip_from, dn_map, wfa_map, wp):
        ev = np.zeros((nd, nf, npp))
        for ip2 in range(npos):
            p = Ptrans[ip_from, ip2]
            if p <= 0:
                continue
            ev += p * _batch_interp(V[c][ip2][dn_map[ip2]], Wf_g, wfa_map[ip2], Wp_g, wp)
        return ev

    policy_leave, policy_return = {}, {}
    for t in reversed(ages[:-1]):
        Vn = {c: np.empty((npos, nd, nf, npp)) for c in countries}
        lg = np.zeros((npos, nd, nf, npp), dtype=bool)
        rg = np.zeros((npos, nd, nf, npp), dtype=bool)
        for c in countries:
            other = destination if c == origin else origin
            A_here = dyn.amenity if c == origin else 0.0
            for ip in range(npos):
                owner = ip >= H[c]["thr"]
                cons, sf, sp = _tenure_flow(ip, H[c], real_c[c][ip], contrib_payg_c[c][ip],
                                            contrib_funded_c[c][ip], rent[c], sat, nd, dvec)
                wf_n = Wf_g[None, :] * (1 + r_f[c]) + sf[:, None]        # (nd,nf)
                wp_n = Wp_g * (1 + r_p) + sp
                dn_map, wfa_map = _transition_maps(ip, wf_n, H[c], netsaleA[c], nd, Dcap, npos, H[c]["down"])
                stay = _crra((cons[:, None, None] + A_here) / cref, g) + beta * cont(c, ip, dn_map, wfa_map, wp_n)
                best = stay
                if t <= dyn.age_mig_max:
                    # le migrant NE capitalise PAS dans c : base = patrimoine DÉBUT de période (Wf_g),
                    # il réalise l'équité (vente exonérée), paie kappa, puis croît à r_f[other]
                    sale = netsaleA[c] if owner else np.zeros(nd)        # vente exonérée à la migration
                    wf_sold = Wf_g[None, :] + sale[:, None]              # (nd,nf) : Wf début + équité réalisée
                    cost = dyn.kappa if c == origin else return_cost[t]
                    paid = np.minimum(wf_sold, cost); unf = cost - paid
                    wf_a = wf_sold - paid
                    il = int(land[c][ip]); owner_land = il >= H[other]["thr"]
                    if owner_land:
                        wf_m0 = np.maximum(wf_a - H[other]["down"], 0.0); d_land = 1; hout_o = H[other]["M"]
                    else:
                        wf_m0 = wf_a; d_land = 0; hout_o = rent[other]
                    after_o = real_c[other][il] - hout_o
                    cons_o = min(after_o, sat); surplus_o = max(0.0, after_o - sat)
                    sf_o = surplus_o + contrib_funded_c[other][il]
                    sp_o = contrib_payg_c[other][il]
                    A_o = dyn.amenity if other == origin else 0.0
                    cons_move = np.maximum(cons_o + A_o - unf, floor)    # (nd,nf)
                    wf_m = wf_m0 * (1 + r_f[other]) + sf_o
                    wp_m = Wp_g * (1 + r_p) + sp_o
                    owner_now_land = d_land == 1
                    dn_m, wfa_m = {}, {}
                    for ip2 in range(npos):
                        owner_next = ip2 >= H[other]["thr"]
                        if owner_next:
                            if owner_now_land:
                                dn_m[ip2] = np.full(nd, min(d_land + 1, Dcap)); wfa_m[ip2] = wf_m
                            else:
                                dn_m[ip2] = np.ones(nd, int); wfa_m[ip2] = np.maximum(wf_m - H[other]["down"], 0.0)
                        else:
                            dn_m[ip2] = np.zeros(nd, int)
                            wfa_m[ip2] = wf_m + (netsaleA[other][d_land] if owner_now_land else 0.0)
                    move = _crra(cons_move / cref, g)[:, :, None] + beta * cont(other, il, dn_m, wfa_m, wp_m)
                    best = np.maximum(stay, move)
                    if c == origin:
                        lg[ip] = move > stay
                    else:
                        rg[ip] = move > stay
                Vn[c][ip] = best
        V = Vn
        policy_leave[t] = lg
        policy_return[t] = rg

    # seuil de départ par âge au point représentatif (patrimoine nul, durée ~ ancienneté si proprio)
    thr_o = H[origin]["thr"]
    threshold, leave_idx, return_idx = {}, {}, {}
    for t in ages[:-1]:
        d_typ = min(t - dyn.age_start + 1, Dcap) if med >= thr_o else 0
        lcol = policy_leave[t][:, d_typ, 0, 0]
        rcol = policy_return[t][:, d_typ, 0, 0]
        li = next((i for i, v in enumerate(lcol) if v), None)
        ri = next((i for i in range(npos - 1, -1, -1) if rcol[i]), None)
        threshold[t] = (pos_o[li], real_o[li]) if li is not None else None
        leave_idx[t], return_idx[t] = li, ri

    sim = {"Ptrans": Ptrans, "land": land, "Wf_g": Wf_g, "Wp_g": Wp_g, "r_f": r_f, "r_p": r_p,
           "kappa": dyn.kappa, "return_cost": return_cost, "sat": sat, "floor": floor,
           "nd": nd, "Dcap": Dcap, "dvec": dvec, "npos": npos, "H": H, "netsaleA": netsaleA,
           "real_c": real_c, "contrib_payg_c": contrib_payg_c, "contrib_funded_c": contrib_funded_c, "rent": rent,
           "origin": origin, "destination": destination, "age_mig_max": dyn.age_mig_max,
           "age_start": dyn.age_start, "amenity": dyn.amenity, "cref": cref}
    return BellmanResult(ages=ages, positions=pos_o, real=real_o, threshold=threshold,
                         leave_idx=leave_idx, return_idx=return_idx, transition=Ptrans,
                         policy_leave=policy_leave, policy_return=policy_return,
                         Wf_g=Wf_g, Wp_g=Wp_g, r_f=r_f, r_p=r_p, flows=None,
                         origin=origin, destination=destination, age_mig_max=dyn.age_mig_max,
                         kappa=dyn.kappa, return_cost=return_cost, role_ratio=role_ratio,
                         land_o2d=o2d, land_d2o=d2o, pop_mass=pop_mass,
                         p_move=p_move, sigma_impl=sigma_impl, cref=cref,
                         ownership=ownership, nd=nd, sim=sim)


class BellmanResult:
    """Sortie du solveur : seuils de départ/retour par âge, transition, cartes et payload de simulation."""

    def __init__(self, ages, positions, real, threshold, leave_idx, return_idx, transition,
                 policy_leave=None, policy_return=None, Wf_g=None, Wp_g=None, r_f=None, r_p=None,
                 flows=None, origin=None, destination=None, age_mig_max=None,
                 kappa=None, return_cost=None, role_ratio=None, land_o2d=None, land_d2o=None,
                 pop_mass=None, p_move=None, sigma_impl=None, cref=None,
                 ownership=None, nd=None, sim=None):
        self.ages = ages
        self.positions = positions
        self.real = real
        self.threshold = threshold          # age -> (position, revenu réel) du seuil de départ
        self.leave_idx = leave_idx
        self.return_idx = return_idx
        self.transition = transition        # matrice de transition de position
        self.policy_leave = policy_leave    # age -> (npos, nd, nf, npp) bool
        self.policy_return = policy_return
        self.Wf_g, self.Wp_g = Wf_g, Wp_g
        self.r_f, self.r_p = r_f, r_p
        self.flows = flows
        self.origin, self.destination = origin, destination
        self.age_mig_max = age_mig_max
        self.kappa = kappa
        self.return_cost = return_cost
        self.role_ratio = role_ratio
        self.land_o2d = land_o2d
        self.land_d2o = land_d2o
        self.pop_mass = pop_mass            # masse de population par ancre (Voronoi) = stationnaire
        self.p_move = p_move
        self.sigma_impl = sigma_impl        # choc permanent implicite de la transition (vs D11)
        self.cref = cref                    # consommation de référence (médiane dispo origine)
        self.ownership = ownership          # canal propriété actif (D4)
        self.nd = nd                        # nombre de durées de détention (axe entier)
        self.sim = sim                      # payload pour la simulation de cohorte (joint d)
