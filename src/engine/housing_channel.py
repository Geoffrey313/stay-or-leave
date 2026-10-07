"""Canal propriété (bloc D4) : mécanique data-indépendante de l'accession et de l'équité.

Option 1 (forme réduite déclarée) : l'accession est DÉTERMINISTE, déclenchée au franchissement
d'un seuil de revenu y_immo (via le taux d'endettement maximal), pas une action discrète dans la
Bellman. Le propriétaire rembourse un prêt dont le PRINCIPAL remboursé est une épargne forcée qui
bâtit un actif terminal (l'équité W_h) ; les INTÉRÊTS sont le coût portant du logement. À la
migration, la résidence est vendue : W_h devient du cash net de frais de transaction, versé dans
le pot funded, ce qui matérialise la RÉTENTION (partir oblige à liquider un actif illiquide).

Garde-fous (cf. discussion D4) :
- Loyer imputé = DIFFÉRENTIEL contre la baseline locataire, jamais un ajout absolu. Le logement
  est déjà dans le coût de la vie (IC_p, bloc D3) via le loyer de référence. Le propriétaire ne
  gagne que l'écart de flux : il paie les intérêts (coût portant) au lieu du loyer de marché, et
  son principal remboursé quitte la consommation pour l'équité. On ne ré-ajoute aucun bénéfice de
  propriété en absolu (sinon double-compte du logement, piège TVA).
- Vente de résidence principale EXONÉRÉE de plus-value (France : totale ; US : abattement large).
  On n'applique PAS l'impôt du capital de D6 au produit de vente du logement.

Ce module ne contient que les formules (amortissement, accession, équité, vente). Le câblage dans
l'état à trois accumulateurs (W_f, W_p, W_h) et les paramètres sourcés vivent ailleurs.
"""
from __future__ import annotations

import numpy as np


def monthly_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Mensualité d'un prêt amortissable à taux fixe (annuité constante)."""
    r = annual_rate / 12.0
    n = term_years * 12
    if abs(r) < 1e-12:
        return principal / n
    return principal * r / (1.0 - (1.0 + r) ** (-n))


def affordable_price(annual_income: float, dti_max: float, annual_rate: float,
                     term_years: int, down_payment_share: float) -> float:
    """Prix de logement le plus élevé finançable : la mensualité ne dépasse pas dti_max * revenu
    mensuel, et l'apport couvre down_payment_share du prix. Prix = principal / (1 - apport)."""
    max_monthly = dti_max * annual_income / 12.0
    r = annual_rate / 12.0
    n = term_years * 12
    principal = max_monthly * n if abs(r) < 1e-12 else max_monthly * (1.0 - (1.0 + r) ** (-n)) / r
    return principal / max(1e-9, 1.0 - down_payment_share)


def equity_schedule(principal: float, annual_rate: float, term_years: int,
                    down_payment: float = 0.0) -> np.ndarray:
    """Équité (capital détenu) par ANNÉE de détention, de l'achat (apport) à l'extinction du prêt.
    equity[k] = apport + principal cumulé remboursé à la fin de l'année k. Hors appréciation du
    bien (posée nulle en réel au premier passage ; à raffiner). Longueur term_years + 1."""
    r = annual_rate / 12.0
    pay = monthly_payment(principal, annual_rate, term_years)
    bal = principal
    eq = [down_payment]
    for _year in range(term_years):
        for _m in range(12):
            interest = bal * r
            bal = max(0.0, bal - (pay - interest))
        eq.append(down_payment + (principal - bal))
    return np.array(eq)


def annual_interest(principal: float, annual_rate: float, term_years: int) -> np.ndarray:
    """Intérêts payés par ANNÉE de détention (coût portant du logement, le flux perdu)."""
    r = annual_rate / 12.0
    pay = monthly_payment(principal, annual_rate, term_years)
    bal = principal
    out = []
    for _year in range(term_years):
        yr_int = 0.0
        for _m in range(12):
            interest = bal * r
            yr_int += interest
            bal = max(0.0, bal - (pay - interest))
        out.append(yr_int)
    return np.array(out)


def owner_flow_differential(market_rent: float, interest_year: float) -> float:
    """Écart de flux de consommation propriétaire vs locataire (garde-fou : DIFFÉRENTIEL).
    Le locataire paie market_rent ; le propriétaire paie les intérêts (coût portant) à la place,
    son principal partant en épargne forcée (équité, hors consommation). Positif si les intérêts
    sont sous le loyer de marché. Aucun bénéfice absolu ajouté."""
    return market_rent - interest_year


def net_sale_proceeds(equity: float, transaction_fee_share: float) -> float:
    """Produit net de la vente de la résidence à la migration : équité moins frais de transaction.
    EXONÉRÉ de plus-value (résidence principale) : aucun impôt du capital de D6 appliqué ici."""
    return max(0.0, equity) * (1.0 - transaction_fee_share)
