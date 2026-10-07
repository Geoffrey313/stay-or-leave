"""Export des DONNÉES de figures en .dat (research-repo-rules §4). Les figures sont tracées en
LaTeX (pgfplots/TikZ) depuis ces fichiers ; ici on ne fait que calculer et écrire les séries, avec
une empreinte de contenu. Données lourdes (solveur) calculées une fois.

Courbes de la Bellman = vraies données : seuil de départ par âge (position au-delà de laquelle
partir domine) et seuil de retour par âge (position en deçà de laquelle rentrer domine), en régime
d'élite sans aménité (là où le départ se produit). Le schéma THÉORIQUE des régions et de leurs bords
est dessiné en TikZ dans le manuscrit, sans données.
"""
from __future__ import annotations

import math

import numpy as np

from src.data.cascade import load_cascade
from src.data.cost_of_living import load_cost_of_living
from src.data.pension import load_pension, load_calibration
from src.data.capital_tax import load_capital_tax
from src.data.housing import load_housing
from src.data.dynamics import load_dynamics
from src.engine.conversion import conversion
from src.engine.bellman import solve as bellman_solve
from src.engine.simulate import simulate_moments
from src.engine.income_process import age_profile, voronoi_masses
from src.analysis.estimate_amenity_migration import estimate as estimate_amenity
from src.figures.style import write_dat

FINE = ["fr", "us", "uk", "ch"]


def _celib(country):
    pts = [p for p in conversion(load_cascade(country)) if p.profile == "celibataire"]
    pts.sort(key=lambda p: p.cost_employer)
    return pts


def _loaders():
    return (load_cost_of_living(), load_pension(), load_capital_tax(),
            load_housing(), load_calibration(), load_dynamics())


def export_all(full: bool = False) -> dict:
    col, pen, tax, house, cal, dyn = _loaders()
    h = {}
    labels = [p.position for p in _celib("fr")]
    xi = list(range(len(labels)))

    # ---- A1 : taux de conversion theta par percentile (%), 4 pays
    theta = {c: [p.theta_brut * 100 for p in _celib(c)] for c in FINE}
    h["A1_theta"] = write_dat("A1_theta", ["i", "fr", "us", "uk", "ch"],
                              [(xi[k], theta["fr"][k], theta["us"][k], theta["uk"][k], theta["ch"][k])
                               for k in xi])
    # ---- A2 : distribution des gains (cout employeur / mediane), 4 pays
    norm = {}
    for c in FINE:
        pts = _celib(c); cost = np.array([p.cost_employer for p in pts], float)
        med = cost[labels.index("P50")]
        norm[c] = cost / med
    h["A2_earnings"] = write_dat("A2_earnings", ["i", "fr", "us", "uk", "ch"],
                                 [(xi[k], norm["fr"][k], norm["us"][k], norm["uk"][k], norm["ch"][k])
                                  for k in xi])
    # ---- A3 : masses de population (Voronoi)
    mass = voronoi_masses(labels)
    h["A3_masses"] = write_dat("A3_masses", ["i", "mass"], [(xi[k], float(mass[k])) for k in xi])
    # ---- A4 : profil age-revenu g(t)
    ages = np.arange(dyn.age_start, dyn.age_retire + 1)
    g = age_profile(ages)
    h["A4_age"] = write_dat("A4_age", ["age", "g"], [(int(ages[k]), float(g[k])) for k in range(len(ages))])
    # ---- A5 : part capitalisee de la retraite, 4 pays
    fund = {}
    for c in FINE:
        pts = _celib(c); ppp = col[c].ppp_aic_local_per_intl_usd; s = []
        for p in pts:
            py, fu = pen[c].split(p.cost_employer, ppp); tot = py + fu
            s.append(fu / tot if tot > 0 else 0.0)
        fund[c] = s
    h["A5_pension"] = write_dat("A5_pension", ["i", "fr", "us", "uk", "ch"],
                                [(xi[k], fund["fr"][k], fund["us"][k], fund["uk"][k], fund["ch"][k])
                                 for k in xi])

    # ---- régime d'élite (A=0) : régions de politique (grilles) et départs par âge
    dyn_dep = dyn.model_copy(update={"amenity": 0.0, "kappa": 20000.0})
    res = bellman_solve("fr", "us", col, pen, tax, house, cal, dyn_dep, role_ratio=1.27)

    def _grid_rows(polmap):
        rows = []
        for a in res.ages:
            pol = polmap.get(a)
            frac = pol.mean(axis=(1, 2, 3)) if pol is not None else np.zeros(len(labels))
            for pi in range(len(labels)):
                rows.append((a, pi, float(frac[pi])))
        return rows

    h["B7_policy"] = write_dat("B7_policy", ["age", "pos", "p"], _grid_rows(res.policy_leave))
    h["B8_return"] = write_dat("B8_return", ["age", "pos", "p"], _grid_rows(res.policy_return))
    h["_grid_ages"] = len(res.ages)
    sim = simulate_moments(res)
    dep = sim["departures_by_age"]
    h["B11_departures"] = write_dat("B11_departures", ["age", "dep"],
                                    [(a, dep[a]) for a in sorted(dep)])
    cref = res.cref

    if full:
        # ---- B12 : net FR->CH vs amenite
        A_grid = np.linspace(0, 0.9 * cref, 10)
        rows = []
        for A in A_grid:
            d = dyn.model_copy(update={"amenity": float(A), "kappa": 20000.0})
            net = simulate_moments(bellman_solve("fr", "ch", col, pen, tax, house, cal, d))["abroad_share_end"]
            rows.append((A / cref, net))
        h["B12_comparative"] = write_dat("B12_comparative", ["aref", "net"], rows)
        # ---- C14 : encadrement net FR->US vs kappa, propriete OFF/ON
        K_grid = np.linspace(5000, 60000, 8)
        rows = []
        for K in K_grid:
            d = dyn.model_copy(update={"amenity": 0.0, "kappa": float(K)})
            off = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, d, ownership=False))["abroad_share_end"]
            on = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, d, ownership=True))["abroad_share_end"]
            rows.append((K / 1000.0, off, on))
        h["C14_encadrement"] = write_dat("C14_encadrement", ["kappa", "off", "on"], rows)
        # ---- B13 : surface SMM distance(A, kappa)
        est = estimate_amenity(destination="us")
        surf = est["surface"]
        A_vals = sorted({r[0] for r in surf}); K_vals = sorted({r[1] for r in surf})
        rows = []
        for a in A_vals:
            for k in K_vals:
                dist = next((r[2] for r in surf if r[0] == a and r[1] == k), math.nan)
                rows.append((a / cref, k / 1000.0, float(dist)))
        h["B13_smm"] = write_dat("B13_smm", ["aref", "kappa", "dist"], rows)
        h["_B13_cols"] = len(K_vals)
        # ---- D16 : force de rappel (net a A=0) + regime par destination
        rows = []
        for dd, reg in (("ch", "over"), ("us", "fit"), ("uk", "under")):
            d = dyn.model_copy(update={"amenity": 0.0, "kappa": 20000.0})
            pull = simulate_moments(bellman_solve("fr", dd, col, pen, tax, house, cal, d))["abroad_share_end"]
            rows.append((dd, pull, reg))
        h["D16_migration"] = write_dat("D16_migration", ["dest", "pull", "regime"], rows)
    return h
