"""Simulation de cohorte sur la distribution jointe (position, durée de détention d, W_f, W_p).

Propage la masse via les grilles de politique du solveur, en MIROIR EXACT des transitions du DP
(bloc D4) : tenure déterministe par rang ; durée d en pur décalage d'indice ; achat (apport sur
W_f, d=1), vente intra-pays ou à la MIGRATION (équité nette de frais sur W_f, exonérée, d=0) ;
le migrant ne capitalise pas dans le pays de départ (base = patrimoine début de période). Les
transitions de patrimoine se font par snap au noeud le plus proche (déterministe). Moments :
taux de départ par âge (et pic), part nette à l'étranger, taux de retour.
"""
from __future__ import annotations

import numpy as np

from src.engine.bellman import _tenure_flow


def _nearest(grid, vals):
    return np.argmin(np.abs(grid[None, :] - np.asarray(vals)[:, None]), axis=1)


def _route(Dn_c, from_pos, from_d, wf, wp, mass, S, thr, down, netsaleA):
    """Distribue `mass` (nf,npp) depuis (from_pos, from_d, patrimoine wf/wp) vers Dn_c via la
    transition de position Ptrans[from_pos] et la transition de tenure (d en index pur), snap grille."""
    Ptrans, Wf_g, Wp_g, Dcap, npos = S["Ptrans"], S["Wf_g"], S["Wp_g"], S["Dcap"], S["npos"]
    owner_now = from_pos >= thr
    tj = _nearest(Wp_g, wp)                                   # (npp,) cible W_p (indép. de ip2)
    for ip2 in range(npos):
        p = Ptrans[from_pos, ip2]
        if p <= 0:
            continue
        owner_next = ip2 >= thr
        if owner_next:
            if owner_now:
                dn = min(from_d + 1, Dcap); wfa = wf
            else:
                dn = 1; wfa = np.maximum(wf - down, 0.0)
        else:
            dn = 0
            wfa = wf + (netsaleA[from_d] if owner_now else 0.0)
        ti = _nearest(Wf_g, wfa)                              # (nf,)
        tgt = Dn_c[ip2, dn]
        pm = p * mass
        for i in range(len(ti)):
            np.add.at(tgt, (np.full(len(tj), ti[i]), tj), pm[i, :])


def simulate_moments(res) -> dict:
    """Moments simulés par propagation de la distribution jointe (position, d, W_f, W_p)."""
    S = res.sim
    o, d_ = S["origin"], S["destination"]
    npos, nd = S["npos"], S["nd"]
    nf, npp = len(S["Wf_g"]), len(S["Wp_g"])
    Wf_g, Wp_g = S["Wf_g"], S["Wp_g"]
    H, netsaleA, rent = S["H"], S["netsaleA"], S["rent"]
    real_c, contrib_payg_c, contrib_funded_c = S["real_c"], S["contrib_payg_c"], S["contrib_funded_c"]
    r_f, r_p, sat = S["r_f"], S["r_p"], S["sat"]
    dvec = S["dvec"]; land = S["land"]
    ages = res.ages

    D = {o: np.zeros((npos, nd, nf, npp)), d_: np.zeros((npos, nd, nf, npp))}
    D[o][:, 0, 0, 0] = res.pop_mass                           # entrée : locataires, patrimoine nul

    # pré-calcul des flux d'épargne par (pays, ip) : sf(nd), sp
    flow = {}
    for c in (o, d_):
        for ip in range(npos):
            _, sf, sp = _tenure_flow(ip, H[c], real_c[c][ip], contrib_payg_c[c][ip],
                                     contrib_funded_c[c][ip], rent[c], sat, nd, dvec)
            flow[(c, ip)] = (sf, sp)

    departures, returns = {}, {}
    for t in ages[:-1]:
        Dn = {o: np.zeros((npos, nd, nf, npp)), d_: np.zeros((npos, nd, nf, npp))}
        dep = ret = 0.0
        for c in (o, d_):
            other = d_ if c == o else o
            pol_grid = res.policy_leave if c == o else res.policy_return
            pol_t = pol_grid.get(t)
            thr_c, thr_o = H[c]["thr"], H[other]["thr"]
            for ip in range(npos):
                owner = ip >= thr_c
                sf, sp = flow[(c, ip)]
                for dd in range(nd):
                    mass = D[c][ip, dd]
                    if mass.sum() <= 0:
                        continue
                    can_move = (pol_t is not None) and (t <= S["age_mig_max"])
                    movep = pol_t[ip, dd] if can_move else np.zeros((nf, npp), bool)
                    stay_mass = mass * (~movep)
                    move_mass = mass * movep
                    # RESTER : capitalise dans c puis transition de tenure
                    if stay_mass.sum() > 0:
                        wf_n = Wf_g * (1 + r_f[c]) + sf[dd]
                        wp_n = Wp_g * (1 + r_p) + sp
                        _route(Dn[c], ip, dd, wf_n, wp_n, stay_mass, S, thr_c, H[c]["down"], netsaleA[c])
                    # PARTIR : base = patrimoine début de période, vente exonérée, kappa, atterrissage
                    if move_mass.sum() > 0:
                        m = move_mass.sum()
                        if c == o:
                            dep += m
                        else:
                            ret += m
                        sale = netsaleA[c][dd] if owner else 0.0
                        wf_sold = Wf_g + sale
                        cost = S["kappa"] if c == o else S["return_cost"][t]
                        paid = np.minimum(wf_sold, cost)
                        wf_a = wf_sold - paid
                        il = int(land[c][ip]); owner_land = il >= thr_o
                        if owner_land:
                            wf_m0 = np.maximum(wf_a - H[other]["down"], 0.0); d_land = 1
                        else:
                            wf_m0 = wf_a; d_land = 0
                        sf_o, sp_o = flow[(other, il)]
                        wf_m = wf_m0 * (1 + r_f[other]) + sf_o[d_land]
                        wp_m = Wp_g * (1 + r_p) + sp_o
                        _route(Dn[other], il, d_land, wf_m, wp_m, move_mass, S,
                               thr_o, H[other]["down"], netsaleA[other])
        D = Dn
        departures[t], returns[t] = dep, ret

    total_dep, total_ret = sum(departures.values()), sum(returns.values())
    peak = max(departures, key=lambda a: departures[a]) if total_dep > 0 else None
    return {"departures_by_age": departures, "peak_departure_age": peak,
            "emigrant_share": total_dep, "return_rate": (total_ret / total_dep) if total_dep > 0 else 0.0,
            "abroad_share_end": float(D[d_].sum())}
