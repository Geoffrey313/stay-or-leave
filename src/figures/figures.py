"""Figures du papier, générées par code (research-repo-rules §4). Chaque figure calcule ses
données depuis le moteur, les trace en EN et FR, et renvoie une empreinte de contenu. Les données
lourdes (solveur) sont calculées une fois puis rendues dans les deux langues.

Régime montré indiqué dans la légende de chaque figure du manuscrit. Palette et style : style.py.
"""
from __future__ import annotations

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
from src.figures import style as S

FINE = ["fr", "us", "uk", "ch"]


def _celib(country):
    pts = [p for p in conversion(load_cascade(country)) if p.profile == "celibataire"]
    pts.sort(key=lambda p: p.cost_employer)
    return pts


def _loaders():
    return (load_cost_of_living(), load_pension(), load_capital_tax(),
            load_housing(), load_calibration(), load_dynamics())


def _label_end(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="bold")


# ------------------------------------------------------------------ A : données
def A1_theta(lang):
    import matplotlib.pyplot as plt
    xs, series = None, {}
    for c in FINE:
        pts = _celib(c); xs = [p.position for p in pts]
        series[c] = np.array([p.theta_brut * 100 for p in pts])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xi = np.arange(len(xs))
    for c in FINE:
        ax.plot(xi, series[c], color=S.COUNTRY_COLOR[c], marker=S.COUNTRY_MARKER[c],
                markersize=5, linewidth=2, label=S.COUNTRY_NAME[lang][c])
        _label_end(ax, xi[-1], series[c][-1], S.COUNTRY_NAME[lang][c], S.COUNTRY_COLOR[c])
    ax.set_xticks(xi); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(S.L[lang]["percentile"]); ax.set_ylabel(S.L[lang]["theta"])
    ax.set_xlim(-0.3, len(xs) - 0.3 + 1.2)
    return S.save(fig, "A1_theta", lang, S.data_hash(*[series[c] for c in FINE]))


def A2_earnings(lang):
    import matplotlib.pyplot as plt
    xs, series = None, {}
    for c in FINE:
        pts = _celib(c); xs = [p.position for p in pts]
        cost = np.array([p.cost_employer for p in pts], float)
        med = cost[[p.position for p in pts].index("P50")] if "P50" in xs else np.median(cost)
        series[c] = cost / med                      # normalisé à la médiane (sans unité, comparable)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xi = np.arange(len(xs))
    for c in FINE:
        ax.plot(xi, series[c], color=S.COUNTRY_COLOR[c], marker=S.COUNTRY_MARKER[c],
                markersize=5, linewidth=2, label=S.COUNTRY_NAME[lang][c])
        _label_end(ax, xi[-1], series[c][-1], S.COUNTRY_NAME[lang][c], S.COUNTRY_COLOR[c])
    ax.set_yscale("log"); ax.set_xticks(xi); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(S.L[lang]["percentile"])
    ax.set_ylabel("Coût employeur / médiane" if lang == "fr" else "Employer cost / median")
    ax.set_xlim(-0.3, len(xs) - 0.3 + 1.2)
    return S.save(fig, "A2_earnings", lang, S.data_hash(*[series[c] for c in FINE]))


def A3_masses(lang):
    import matplotlib.pyplot as plt
    pts = _celib("fr"); labels = [p.position for p in pts]
    mass = voronoi_masses(labels)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xi = np.arange(len(labels))
    ax.bar(xi, mass, color=S.SEQ_HUE, width=0.72)
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(S.L[lang]["percentile"]); ax.set_ylabel(S.L[lang]["mass"])
    return S.save(fig, "A3_masses", lang, S.data_hash(mass))


def A4_age_profile(lang):
    import matplotlib.pyplot as plt
    dyn = load_dynamics()
    ages = np.arange(dyn.age_start, dyn.age_retire + 1)
    g = age_profile(ages)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(ages, g, color=S.SEQ_HUE, linewidth=2)
    ax.set_xlabel(S.L[lang]["age"]); ax.set_ylabel(S.L[lang]["income_index"])
    return S.save(fig, "A4_age_profile", lang, S.data_hash(g))


def A5_pension(lang):
    import matplotlib.pyplot as plt
    col, pen, *_ = _loaders()
    xs, series = None, {}
    for c in FINE:
        pts = _celib(c); xs = [p.position for p in pts]
        ppp = col[c].ppp_aic_local_per_intl_usd
        share = []
        for p in pts:
            payg, funded = pen[c].split(p.cost_employer, ppp)
            tot = payg + funded
            share.append(funded / tot if tot > 0 else 0.0)
        series[c] = np.array(share)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xi = np.arange(len(xs))
    for c in FINE:
        ax.plot(xi, series[c], color=S.COUNTRY_COLOR[c], marker=S.COUNTRY_MARKER[c],
                markersize=5, linewidth=2, label=S.COUNTRY_NAME[lang][c])
    ax.set_xticks(xi); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(S.L[lang]["percentile"]); ax.set_ylabel(S.L[lang]["share_funded"])
    ax.set_ylim(-0.05, 1.03)
    # France et États-Unis coïncident à zéro (aucun pilier capitalisé obligatoire), Royaume-Uni et
    # Suisse se superposent : légende plutôt qu'étiquettes directes (qui se chevaucheraient).
    ax.legend(frameon=False, fontsize=8, loc="center left", ncol=2)
    return S.save(fig, "A5_pension", lang, S.data_hash(*[series[c] for c in FINE]))


# ------------------------------------------------------------------ B : dynamique
def B7_policy(lang, data):
    import matplotlib.pyplot as plt
    ages, labels, grid = data["ages"], data["labels"], data["leave_grid"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(grid.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1,
                   extent=[ages[0], ages[-1], -0.5, len(labels) - 0.5])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(S.L[lang]["age"]); ax.set_ylabel(S.L[lang]["pos"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(partir)" if lang == "fr" else "P(leave)", fontsize=9)
    ax.grid(False)
    return S.save(fig, "B7_policy", lang, S.data_hash(grid))


def B8_return(lang, data):
    import matplotlib.pyplot as plt
    ages, labels, grid = data["ages"], data["labels"], data["return_grid"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(grid.T, aspect="auto", origin="lower", cmap="Oranges", vmin=0, vmax=1,
                   extent=[ages[0], ages[-1], -0.5, len(labels) - 0.5])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(S.L[lang]["age"]); ax.set_ylabel(S.L[lang]["pos"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(rentrer)" if lang == "fr" else "P(return)", fontsize=9)
    ax.grid(False)
    return S.save(fig, "B8_return", lang, S.data_hash(grid))


def B11_departures(lang, data):
    import matplotlib.pyplot as plt
    ages = np.array(sorted(data["dep"].keys()))
    vals = np.array([data["dep"][a] for a in ages])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(ages, vals, color=S.SEQ_HUE, width=0.9)
    if data["peak"] is not None:
        ax.axvline(data["peak"], color=S.COUNTRY_COLOR["fr"], linewidth=1.5, linestyle="--")
        _label_end(ax, data["peak"], vals.max(), ("pic" if lang == "fr" else "peak"), S.COUNTRY_COLOR["fr"])
    ax.set_xlabel(S.L[lang]["age"]); ax.set_ylabel(S.L[lang]["dep_share"])
    return S.save(fig, "B11_departures", lang, S.data_hash(ages, vals))


def B12_comparative(lang, data):
    import matplotlib.pyplot as plt
    A, net = data["A_grid"], data["net_vs_A"]
    cref = data["cref"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(A / cref, net, color=S.COUNTRY_COLOR["ch"], marker="o", markersize=4, linewidth=2)
    ax.axhline(0.06, color=S.MUTED, linewidth=1, linestyle=":")
    _label_end(ax, (A / cref)[-1], net[-1], "FR→CH", S.COUNTRY_COLOR["ch"])
    ax.set_xlabel(S.L[lang]["amenity_cref"]); ax.set_ylabel(S.L[lang]["net_abroad"])
    return S.save(fig, "B12_comparative", lang, S.data_hash(A, net))


def B13_smm_surface(lang, data):
    import matplotlib.pyplot as plt
    Ag, Kg, Z = data["A_axis"], data["K_axis"], data["dist_grid"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.pcolormesh(Ag / data["cref"], Kg / 1000.0, Z.T, cmap="Blues_r", shading="auto")
    j, i = np.unravel_index(np.nanargmin(Z.T), Z.T.shape)
    ax.plot((Ag / data["cref"])[i], (Kg / 1000.0)[j], marker="*", color=S.COUNTRY_COLOR["fr"], markersize=14)
    ax.set_xlabel(S.L[lang]["amenity_cref"]); ax.set_ylabel(S.L[lang]["kappa"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(S.L[lang]["distance"], fontsize=9)
    return S.save(fig, "B13_smm_surface", lang, S.data_hash(Ag, Kg, Z))


# ------------------------------------------------------------------ C : bords
def C14_encadrement(lang, data):
    import matplotlib.pyplot as plt
    K, off, on = data["kappa_grid"], data["net_off"], data["net_on"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(K / 1000.0, off, color=S.COUNTRY_COLOR["us"], marker="s", markersize=4, linewidth=2,
            label=S.L[lang]["without_housing"])
    ax.plot(K / 1000.0, on, color=S.COUNTRY_COLOR["ch"], marker="D", markersize=4, linewidth=2,
            label=S.L[lang]["with_housing"])
    ax.axhline(0.06, color=S.MUTED, linewidth=1, linestyle=":")
    _label_end(ax, (K / 1000.0)[-1], 0.06, S.L[lang]["observed"], S.MUTED)
    ax.set_xlabel(S.L[lang]["mig_cost"]); ax.set_ylabel(S.L[lang]["net_abroad"])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return S.save(fig, "C14_encadrement", lang, S.data_hash(K, off, on))


# ------------------------------------------------------------------ D : matrice
def D16_migration(lang, data):
    import matplotlib.pyplot as plt
    dests = data["dests"]; pull = data["pull"]; regime = data["regime"]
    reg_color = {"over": S.COUNTRY_COLOR["ch"], "fit": S.COUNTRY_COLOR["us"], "under": S.COUNTRY_COLOR["uk"]}
    reg_label = {"fr": {"over": "sur-attire", "fit": "s'ajuste", "under": "sous-retenu"},
                 "en": {"over": "over-attracts", "fit": "matched", "under": "under-retained"}}
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, len(dests) + 1)
    y0 = (len(dests) + 1) / 2.0
    ax.scatter([1.2], [y0], s=1400, color=S.COUNTRY_COLOR["fr"], zorder=3)
    ax.text(1.2, y0, "FR", color="white", ha="center", va="center", fontweight="bold", zorder=4)
    for k, d in enumerate(dests):
        y = k + 1
        w = 0.5 + 6.0 * float(pull[d])              # largeur ∝ force de rappel (net à A=0)
        ax.annotate("", xy=(7.4, y), xytext=(1.9, y0),
                    arrowprops=dict(arrowstyle="-|>", color=reg_color[regime[d]], lw=w, alpha=0.85))
        ax.scatter([7.8], [y], s=900, color=reg_color[regime[d]], zorder=3)
        ax.text(7.8, y, d.upper(), color="white", ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(8.5, y, f"{S.COUNTRY_NAME[lang][d]} ({reg_label[lang][regime[d]]})",
                color=S.INK, ha="left", va="center", fontsize=8)
    return S.save(fig, "D16_migration", lang, S.data_hash(np.array([pull[d] for d in dests])))


# ------------------------------------------------------------------ pilote
def _heavy_data(full):
    """Calcule une fois les données lourdes (solveur) partagées par les deux langues."""
    col, pen, tax, house, cal, dyn = _loaders()
    # Régime d'élite SANS aménité (A=0, kappa headline) : c'est là que le départ se produit, donc
    # là où la politique rester/partir et le pic de départs sont lisibles (l'aménité par défaut,
    # ~1,45 c_ref, éteindrait tout départ et donnerait une figure vide).
    dyn_dep = dyn.model_copy(update={"amenity": 0.0, "kappa": 20000.0})
    res_elite = bellman_solve("fr", "us", col, pen, tax, house, cal, dyn_dep, role_ratio=1.27)
    labels = res_elite.positions
    ages = np.array(res_elite.ages)
    # B7 : P(partir) moyenné sur patrimoine et durée, par (âge, position)
    leave_grid = np.zeros((len(ages), len(labels)))
    for ia, a in enumerate(res_elite.ages):
        pol = res_elite.policy_leave.get(a)
        if pol is not None:
            leave_grid[ia] = pol.mean(axis=(1, 2, 3))      # (npos,)
    # B8 : P(rentrer) moyennée sur patrimoine et durée, par (âge, position) à l'étranger
    return_grid = np.zeros((len(ages), len(labels)))
    for ia, a in enumerate(res_elite.ages):
        pol = res_elite.policy_return.get(a)
        if pol is not None:
            return_grid[ia] = pol.mean(axis=(1, 2, 3))
    sim_elite = simulate_moments(res_elite)
    data = {"ages": ages, "labels": labels, "leave_grid": leave_grid, "return_grid": return_grid,
            "dep": sim_elite["departures_by_age"], "peak": sim_elite["peak_departure_age"],
            "cref": res_elite.cref}
    if full:
        # B12 : net FR->CH vs aménité A
        A_grid = np.linspace(0, 0.9 * res_elite.cref, 10)
        net_vs_A = []
        for A in A_grid:
            d = dyn.model_copy(update={"amenity": float(A), "kappa": 20000.0})
            net_vs_A.append(simulate_moments(bellman_solve("fr", "ch", col, pen, tax, house, cal, d))["abroad_share_end"])
        data["A_grid"] = A_grid; data["net_vs_A"] = np.array(net_vs_A)
        # C14 : net FR->US vs kappa, propriété OFF/ON (A=0, régime proxy)
        K_grid = np.linspace(5000, 60000, 8)
        off, on = [], []
        for K in K_grid:
            d = dyn.model_copy(update={"amenity": 0.0, "kappa": float(K)})
            off.append(simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, d, ownership=False))["abroad_share_end"])
            on.append(simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal, d, ownership=True))["abroad_share_end"])
        data["kappa_grid"] = K_grid; data["net_off"] = np.array(off); data["net_on"] = np.array(on)
        # B13 : surface SMM distance(A, kappa), FR->US proxy
        est = estimate_amenity(destination="us")
        surf = est["surface"]
        A_vals = sorted(set(r[0] for r in surf)); K_vals = sorted(set(r[1] for r in surf))
        Z = np.full((len(A_vals), len(K_vals)), np.nan)
        for r in surf:
            Z[A_vals.index(r[0]), K_vals.index(r[1])] = r[2]
        data["A_axis"] = np.array(A_vals, float); data["K_axis"] = np.array(K_vals, float); data["dist_grid"] = Z
        # D16 : force de rappel (net à A=0) + régime par destination
        pull, regime = {}, {"ch": "over", "us": "fit", "uk": "under"}
        for dd in ("us", "uk", "ch"):
            d = dyn.model_copy(update={"amenity": 0.0, "kappa": 20000.0})
            pull[dd] = simulate_moments(bellman_solve("fr", dd, col, pen, tax, house, cal, d))["abroad_share_end"]
        data["dests"] = ["ch", "us", "uk"]; data["pull"] = pull; data["regime"] = regime
    return data


def generate(full: bool = False) -> dict:
    """Génère toutes les figures en EN et FR ; renvoie {nom: empreinte} (contrôle de déterminisme)."""
    S.setup()
    data = _heavy_data(full)
    out = {}
    light = [A1_theta, A2_earnings, A3_masses, A4_age_profile, A5_pension]
    dyn_fig = [("B7_policy", B7_policy), ("B8_return", B8_return), ("B11_departures", B11_departures)]
    heavy = [("B12_comparative", B12_comparative), ("C14_encadrement", C14_encadrement),
             ("B13_smm_surface", B13_smm_surface), ("D16_migration", D16_migration)]
    for lang in ("fr", "en"):
        for fn in light:
            out[f"{fn.__name__}:{lang}"] = fn(lang)
        for _, fn in dyn_fig:
            out[f"{fn.__name__}:{lang}"] = fn(lang, data)
        if full:
            for _, fn in heavy:
                out[f"{fn.__name__}:{lang}"] = fn(lang, data)
    return out
