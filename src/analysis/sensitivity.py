"""Sensibilité au premier ordre des résultats de tête aux paramètres POSÉS (bloc SL-US-03).

On balaye un paramètre de calibration à la fois, autour de sa valeur centrale, et on suit les
métriques de VERDICT, sans ré-estimer : la part nette à l'étranger au régime headline (proxy, A=0,
kappa de référence), la même avec le canal propriété activé (borne d'encadrement), et la part nette
en régime d'élite (role_ratio=1,27). Le verdict tient si, à chaque valeur, le canal propriété
retient (net ON < net OFF) et l'élite part (net élite > net headline). Déterministe, aucune
modification du cœur : la valeur centrale reproduit le headline au bit près (mêmes appels que
reproduce).

Les paramètres sont des champs de `Calibration`, modifiés par `model_copy` ; pour l'aversion,
`r_mkt_certainty_equivalent` étant une propriété calculée ($r_{mkt}-\\tfrac12\\gamma\\sigma^2$), le
balayage de gamma propage à la fois la courbure du flux et le taux équivalent-certain.
"""
from __future__ import annotations

from src.engine.bellman import solve as bellman_solve
from src.engine.simulate import simulate_moments

# nom affiché -> champ de Calibration ; valeurs centrales dans calibration.csv
PARAMS = {
    "gamma": "risk_aversion",
    "r_mkt": "r_mkt_real",
    "saturation": "consumption_saturation_intl_usd",
}


def sensitivity_sweep(col, pen, tax, house, cal, dyn, field: str, values,
                      a_ref: float = 0.0, kappa_ref: float = 20000.0, elite_rr: float = 1.27) -> dict:
    """Balaye `field` (champ de Calibration) sur `values`. Renvoie, par valeur, la part nette à
    l'étranger headline (proxy, locataire), avec propriété, et en régime d'élite ; plus le drapeau
    de verdict (propriété retient ET élite part à toute valeur)."""
    d = dyn.model_copy(update={"amenity": float(a_ref), "kappa": float(kappa_ref)})
    rows, holds = [], True
    for v in values:
        cal_v = cal.model_copy(update={field: v})
        off = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal_v, d, ownership=False))["abroad_share_end"]
        on = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal_v, d, ownership=True))["abroad_share_end"]
        el = simulate_moments(bellman_solve("fr", "us", col, pen, tax, house, cal_v, d, role_ratio=elite_rr, ownership=False))["abroad_share_end"]
        rows.append((float(v), float(off), float(on), float(el)))
        holds = holds and (on < off) and (el > off)
    return {"field": field, "rows": rows, "holds": holds}
