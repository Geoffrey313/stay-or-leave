# Données de reproduction — « Rester ou partir »

Ce dossier documente et versionne les **données transformées** qui alimentent la
chaîne de reproduction (`reproduce.py`). Les données brutes et les scripts qui les
téléchargent restent hors du dépôt (`data-acquisition/`, ignoré par git). Chaque
entrée du modèle est un **paramètre** daté millésime 2026 ; le dépôt est donc une
base de paramètres, pas un panel d'observations.

## Périmètre

Neuf pays, avec granularité infra-nationale là où le modèle la fixe :

| Code | Pays | Juridiction retenue | Villes de référence (rang urbain q) |
|------|------|---------------------|-------------------------------------|
| FR | France | national | Paris (1,0), Lyon (0,7), ville moyenne (0,4) |
| US | États-Unis | Texas (sans impôt d'État) | Austin (0,7) |
| CH | Suisse | canton de Zoug | Bâle / canton (0,7) |
| UK | Royaume-Uni | national | Londres, ville secondaire |
| DE | Allemagne | national | Berlin (0,7) |
| CA | Canada | Ontario | Toronto, ville secondaire |
| IT | Italie | national | Milan, ville secondaire |
| ES | Espagne | Madrid | Madrid |
| JP | Japon | national | Tokyo, ville secondaire |

Unité de compte primitive : le **coût total employeur** annuel C (ce que l'employeur
débourse), pas le brut ni le net.

## Statut de collecte

`manquant` = à collecter · `partiel` = valeurs illustratives dans le framework, à
sourcer et recalculer · `collecté` = versionné et tracé dans ce dossier.

**Provenance des valeurs actuelles du framework.** Le ledger
`docs/internal/research/audit-sources.md` (notes de travail, hors dépôt) classe chaque
paramètre existant en sourcé / estimé / inventé. Points fermes, à ne pas recollecter :
barèmes socio-fiscaux France 2026, dette étudiante UK/US/FR (OCDE *Education at a Glance*
2025), cadre santé US (plafond de reste à charge, coverage gap, seuil de pauvreté fédéral),
profil salaire-âge INSEE, fiscalité du capital FR et CH. Trou central : θ(C) hors France,
entièrement posé par typologie. Chaque paramètre qui passe d'inventé ou estimé à sourcé
reçoit sa référence datée dans ce ledger.

## Table des besoins de données

### D1 — Cascade socio-fiscale θ(C) : coût employeur → pouvoir d'achat

**Ancrage data-motivé sur la distribution des gains.** θ n'est pas évalué sur une grille de
montants ronds mais sur des positions de la distribution des salaires (percentiles P10, P25,
P50, P75, P90, P95, P99, P99,9), position comparable entre pays (comme le rang urbain pour les
villes). Chaque percentile de salaire net est converti en coût employeur par le coin
cotisations patronales propre au pays (pas un multiplicateur plat). Queue (P99,9) extrapolée
par loi de Pareto ajustée sur P90/P95/P99, déclarée modélisée. Faits : France (INSEE 2024, net EQTP privé), US (DQYDJ/Census ASEC 2023, salaire brut, tous
actifs, le bas plus bas car temps partiel inclus), UK (ONS ASHE 2025 P10-P90 temps plein +
ad-hoc 2024 P95/P99 remontés à 2025 par le ratio P90), chacun dans
`data/params/<pays>/earnings_distribution.csv`. Reste : étiquetage en percentiles des six OCDE
et bas (P10/P25) par TaxBEN.

**θ est défini hors TVA.** La cascade θ s'arrête au revenu disponible rapporté au coût
employeur. La taxe sur la consommation n'entre pas ici : elle vit une seule fois dans le
bloc D3 (indice IC_p), dont les niveaux de prix comparatifs sont déjà TVA incluse. La
mettre aussi dans θ la compterait deux fois. Deux objets : θ brut (salaire seul) et θ
élargi (prestations incluses).

**Stratégie (validée) : hybride.** Microsimulation où elle existe (OpenFisca-France,
OpenFisca-US / TAXSIM pour le fédéral + Texas, PolicyEngine-UK), moteur EUROMOD pour
DE/IT/ES (accès licence JRC sur inscription ; repli OCDE + barèmes nationaux le temps
de l'obtenir), et barèmes statutaires 2026 codés à la main pour CH (canton de Zoug),
CA (Ontario), JP et pour la queue haute (> 250 % du salaire moyen) que les moteurs
plafonnent. Le détail par pays est dans `data/sources.yaml` (bloc `tax_engines`).


| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Cotisations patronales (barème, assiettes, plafonds) | — | pays/juridiction, 2026 | OCDE *Taxing Wages* 2026 | Autorités nationales (FR : URSSAF, PLFSS 2026) ; PwC/EY *Worldwide Tax Summaries* | OCDE SDMX ; portails nationaux | partiel |
| Cascade θ(C) fine : **France** (OpenFisca-France), **US-Texas** et **UK** (PolicyEngine) | θ brut, θ élargi | 3 profils, grille 20k–400k | moteurs de microsimulation | — | `data/params/<pays>/cascade.csv` | collecté (FR, US, UK) |
| Cascade θ(C) grossière : **DE, IT, ES, CH, CA, JP** (OCDE Taxing Wages, national) | θ élargi faisant autorité | célibataire 3 pts (67/100/167% SM) + couple 1 pt | OCDE *Taxing Wages* (SDMX, montants LC/AV_TW) | — | `data/params/<pays>/cascade.csv` | collecté (DE, IT, ES, CH, CA, JP) |
| Cotisations salariales | — | pays, 2026 | OCDE *Taxing Wages* 2026 | idem national | SDMX | partiel |
| Barème d'impôt sur le revenu (tranches, décote, quotient) | — | pays + État/canton, 2026 | Autorités fiscales nationales | OCDE *Tax Database* ; PwC/KPMG/EY | portails ; PDF barèmes | partiel |
| Impôt infra-national (État/canton/région) | — | TX, Zoug, Ontario, Madrid | Administrations locales (ESTV Zoug, CRA Ontario, Agencia Madrid) | KPMG cantons suisses ; Tax Foundation (US) | portails | manquant |
| Taxe sur la consommation (TVA / sales tax, taux effectif panier) | — | pays/juridiction, 2026 | OCDE *Consumption Tax Trends* | Commission UE *VAT rates* ; Tax Foundation (TX sales tax) | SDMX ; PDF | partiel |

### D2 — Zone de transferts (sous le salaire minimum)

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Salaire minimum | $\underline{y}$ | pays, 2026 | OCDE *Minimum Wages* ; Eurostat | Autorités nationales (SMIC, NLW, Mindestlohn, SMI) | SDMX | partiel |
| Revenu minimum garanti | $\tau^{+}$ | pays, 2026 | OCDE *TaxBEN* (modèle prestations) | MISSOC (UE) ; agences nationales | tableurs TaxBEN | manquant |
| Complément d'activité / in-work benefit | $\tau^{+}$ | pays | OCDE *TaxBEN* | FR : prime d'activité ; US : EITC ; UK : Universal Credit | tableurs | manquant |
| Aide au logement | $\tau^{+}$ | pays | OCDE *TaxBEN* | FR : APL (CNAF) ; MISSOC | tableurs | manquant |
| Prestations sous condition de ressources (extinction) | $b_k(C)$ | pays | OCDE *TaxBEN* | agences nationales | tableurs | manquant |

### D3 — Coût de la vie et pouvoir d'achat réel

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Parité de pouvoir d'achat consommation (TTC) | $\mathrm{IC}_p$ | 9 pays, 2024 | OCDE *PPP* (AIC, monnaie locale/$ int.) | Eurostat PLI | SDMX | collecté (`cost_of_living.csv`) |
| Taxe sur la consommation (décompose IC_p, pas θ) | — | pays | OCDE *Consumption Tax Trends* (VAT Revenue Ratio) | INSEE comptes des ménages | SDMX ; web | partiel (FR posé [E]) |
| Indice coût de la vie par ville | $\mathrm{IC}_p(q)$ | ville (540+) | Numbeo *Cost of Living* (pages web publiques : `rankings.jsp`, `/cost-of-living/in/<ville>`) | Mercer / ECA | web public gratuit (API = option payante) | manquant |

### D4 — Immobilier comme flux capitalisable

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Prix au m² par ville / rang urbain | $P_{m^2}(p,q)$ | ville | Numbeo *Property Investment* (`property-investment/rankings.jsp`, profils par ville) | Deloitte *Property Index* ; stats nationales (INSEE/Notaires, Destatis) | web public gratuit ; PDF | manquant |
| Loyer au m² par ville | $\ell_p(C,q)$ | ville | Numbeo *Rent* (profils par ville : loyer 1 et 3 pièces, centre/périphérie) | idem national | web public gratuit | manquant |
| Taux de crédit immobilier | $r_m$ | pays, 2026 | Banques centrales (Banque de France, Freddie Mac US, BoE, Bundesbank) | OCDE *Analytical House Prices* | FRED ; SDMX | manquant |
| Taux d'endettement maximal admissible | $\bar{\eta}$ | pays | Régulateurs (HCSF France ~35 %) | normes bancaires nationales | doc | partiel |
| Frais de transaction / portage | — | pays | Notaires / barèmes nationaux | Global Property Guide | web | manquant |

### D5 — Système de retraite et bascule charge → actif

**Première brique posée (`pension.csv`, `calibration.csv`, `src/engine/pension.py`).** Régime
par pays (répartition FR/DE/IT/ES/JP, capitalisation US/UK/CH/CA, classification du cadre),
part de cotisation retraite (FR 27,8 % du brut OCDE, US OASDI 12,4 %, UK auto-enrolment 8 %),
r_rep/r_mkt/σ/γ/horizon en calibration déclarée (statut par valeur). Gain de bascule = flux retraite
revalorisé de r_rep à r_mkt sur l'horizon. **r_mkt est une espérance d'actif RISQUÉ, pas un certain** :
le gain brut est une borne haute avant prime de risque, pas « l'avantage ». Π transporte r_mkt ET sa
variance σ² ; l'ajustement de risque canonique est fait une fois dans la Bellman. L'équivalent-certain
illustratif (taux CRRA/Merton r_mkt−½γσ² ≈ 3,1 %) vaut ~45 % du brut : l'écart est un résultat, pas une
présentation. Réserves déclarées : r_mkt = moyenne pondérée par le glide path (effet de séquence ignoré) ;
cotisation figée = canal rendement isolé (le taux de cotisation par pays est une marge distincte).
À affiner : parts des six autres pays, r_rep par pays (démographie, D14), extraction par microsimulateur.


| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Taux de remplacement (répartition) | — | pays, par niveau de revenu | OCDE *Pensions at a Glance* 2023/2025 | FR : COR ; national | SDMX ; PDF | partiel |
| Rendement implicite de la répartition | $r_{\mathrm{rep}}(C)$ | pays | Croissance masse salariale = démographie + productivité (D14) | projections nationales (COR) | dérivé | manquant |
| Pilier capitalisé (taux, plafonds : 401k, RRSP, DC, LPP) | — | pays, 2026 | Autorités (IRS, CRA, OFAS LPP) | OCDE *Pensions at a Glance* | portails | manquant |
| Âge de départ / horizon de travail | $T_w$ | pays | OCDE *Pensions at a Glance* | national | SDMX | partiel |
| Rendement de marché (capitalisation) | $r_{\mathrm{mkt}}$ | global | *Global Investment Returns Yearbook* (UBS/DMS) | prime de risque académique | annuaire | partiel |

### D6 — Fiscalité du capital

**Brique posée (`capital_tax.csv`, `src/engine/surplus.py`).** Le surplus réel au-dessus d'un
seuil de saturation de la consommation est investi au rendement net = r_mkt·(1−τ), τ = impôt
sur les plus-values (FR 31,4 % et CH 0 % fermes ; autres standards flaggés). Même traitement du
risque que D5 : brut avant prime de risque + EC illustratif, variance portée vers Π. Surplus nul
sous saturation (~P75), puis divergence rapide au sommet. D5 (retraite capitalisée) + D6 (surplus
capitalisé) = l'avantage patrimonial complet de la capitalisation.

**Π unique assemblé (`src/engine/patrimony.py`).** Valeur patrimoniale lifetime en valeur
présence = consommation réelle actualisée à ρ (plafonnée à la saturation) + stocks terminaux
retraite (D5) et surplus (D6), ramenés à ρ. Deux versions, brut (r_mkt) et équivalent-certain
(r_mkt−½γσ²) ; l'ajustement de risque canonique est à la Bellman. Conventions déclarées : φ=1
(position de destination égale, prime salariale non intégrée, relâchée en D7), baseline locataire
(logement = D4), imposition du capital à l'accrual. Résultat FR→US (célibataire, EC) : rester
domine P10/P25, partir dès P50, gain jusqu'à ~11,6 M$ int. au P99,9 ; FR→UK plus mou, repasse à
rester au P99.


| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Impôt sur les plus-values mobilières | $\tau_k$ | pays, 2026 | OCDE *Tax Database* | PwC/KPMG/EY *Worldwide Tax Summaries* 2026 | SDMX ; web | partiel |
| Imposition des dividendes | — | pays | OCDE *Tax Database* | idem | SDMX | manquant |
| Impôt sur la fortune (portefeuille) | — | pays/canton (CH) | Administrations (ESTV Zoug) | KPMG | portails | partiel |

### D7 — Prime salariale pour rôle comparable

**Brique posée (`wage_premium.csv`, `src/engine/wage_premium.py`) ET résultat de premier ordre.**
φ est une TRANSLATION DE RANG, pas un facteur de niveau : l'écart de niveau salarial est déjà
porté par l'ancrage de chaque cascade sur sa distribution ; réappliquer le ratio de salaire moyen
serait un double-compte. φ répond à « un cadre FR-percentile p atterrit à quel percentile de
destination ? », via la rémunération réelle à rôle comparable (`role_ratio_ppp`). Constat : sous
role_ratio=1 (même rémunération réelle comparable, atterrissage plus bas dans une distribution plus
riche), le gain « partir » FR→US **s'effondre et passe négatif dès P50** (P99,9 : +11,6 M$ à φ=1 →
−1,1 M$). Le « partir gagne au sommet » à φ=1 venait donc du NIVEAU (hypothèse optimiste), pas d'un
vrai avantage de rang. La réponse dépend entièrement de la prime à rôle comparable réelle, à SOURCER
(Eurostat SES par profession, BLS OES, ECA/Mercer) — c'est du premier ordre, pas une sensibilité.
Friction (diplôme, langue, chômage initial) = bloc D8 (κ), jamais dans φ.

**Headline = seuil, pas verdict (`break_even_role_ratio`).** Le résultat se publie comme le seuil
role_ratio* par position/profil : la prime réelle à rôle comparable à partir de laquelle partir
domine. FR→US ~1,00 (P10) à 1,13 (P99,9) ; FR→UK ~1,06 à 1,14 (seuil croissant avec le revenu, UK
moins attractif). « Partir paie SSI la prime réelle dépasse role_ratio* » ; on situe la fourchette
sourcée par rapport au seuil, sans parier sur un scalaire. Hors coût de migration : κ (D8) relèvera
le seuil. Aucun verdict rester/partir tant que role_ratio n'est pas sourcé ou le résultat non
reformulé en seuil. Sourçage à venir : fourchette par profession ISCO, en PPA, avec le caveat que
le gap d'occupation en coupe est une BORNE HAUTE du premium causal (sélection positive des partants ;
κ et A estimés sur flux D13 en absorberont une part).


| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Salaire moyen comparable | $\phi$, $\rho_w$ | pays, 2025–2026 | OCDE *Average Wages* | Eurostat *Structure of Earnings* ; BLS OES (US) | SDMX | partiel |
| Écart de rémunération par métier (cadre) | $\phi$ | pays/profession | ECA International / Mercer *Pay* | enquêtes package expat | payant | manquant |
| Volatilité relative du revenu expatrié | $\sigma_p$ | pays | littérature migration | dérivé enquêtes | — | manquant |

### D8 — Coût de migration et aménité

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Coût fixe d'installation | $\kappa_0$ | pays | barèmes relocation (ECA/Mercer) | devis déménagement international | payant | manquant |
| Friction linguistique | $f^{\mathrm{ling}}$ | paire origine-destination | indices de distance linguistique (Melitz-Toubal) | — | littérature | manquant |
| Risque d'emploi initial (mois) | $n^{\mathrm{mois}}$ | pays | durées de recherche d'emploi (OCDE) | enquêtes expat | SDMX | manquant |
| Aménité « chez soi » | $A$ | — | calibrée par écart seuil statique/dynamique | — | interne au modèle | partiel |

### D9 — Risque santé (reste à charge équivalent-certain)

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Reste à charge moyen | $m^{\mathrm{san}}$ | pays | OCDE *Health Statistics* (OOP) | WHO *Global Health Expenditure* | SDMX | partiel |
| Reste à charge catastrophique (queue) | $m^{\mathrm{san}}$ | pays | OCDE ; Commonwealth Fund | US : KFF / Peterson-KFF, MEPS | web ; API | manquant |
| Dépendance au revenu (US : Medicaid, subventions ACA, coverage gap) | $m^{\mathrm{san}}(C)$ | US par revenu/État | KFF | HealthCare.gov barèmes 2026 | web | manquant |

### D10 — Éducation : coût des études des enfants et dette passée de l'agent

Deux composantes distinctes, toutes deux dans le flux éducation du modèle (PDF §23) : le
coût prospectif des études des enfants, et la dette étudiante passée que l'agent traîne.

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Frais universitaires par enfant (public / privé) | $u^{\mathrm{tui}}$ | pays, 2026 | OCDE *Education at a Glance* 2025 | US : College Board ; UK : plafond £9 250 | SDMX ; web | partiel |
| Coût actualisé des études de n enfants | $e_p(C,n)$ | dérivé | — | — | calculé | manquant |
| Part de diplômés endettés | — | pays | OCDE *Education at a Glance* 2025 | US : Fed ; FR : OVE/Sénat | web | partiel (UK/US/FR sourcés) |
| Dette étudiante médiane (endettés) | — | pays | OCDE *Education at a Glance* 2025 | national | web | partiel (UK/US/FR sourcés) |
| Scénario « aspiration élite » (interpolation revenu) | — | — | hypothèse structurelle | — | posé | inventé, à assumer |

### D11 — Processus de revenu de cycle de vie

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Profil salaire-âge (cadres) | $g(t)$ | FR, par âge | INSEE *base Tous salariés* (ex-DADS) | OCDE ; Eurostat SES | fichiers | partiel |
| Variance du choc permanent | $\sigma_\eta$ | — | estimations académiques revenus permanents | INSEE panel | littérature | partiel |
| Variance du choc transitoire | $\sigma_\varepsilon$ | — | idem | — | littérature | manquant |

### D12 — Paramètres financiers et de préférences

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Facteur d'actualisation (framework : ρ = 2 %) | $\beta$ | — | calibration standard cycle de vie | littérature | — | inventé |
| Aversion au risque (framework : γ = 2) | $\gamma$ | — | littérature (CRRA ~2–5) | — | — | inventé |
| Rendement de marché (framework : 4,5 % réel) | $r_{\mathrm{mkt}}$ | global | *Global Investment Returns Yearbook* | Damodaran | annuaire ; web | inventé |
| Glide path actions/obligations (85 % → 30 %) | — | — | règle target-date (110 − âge) | — | — | estimé |

**Paramètres structurels non observables.** L'aménité $A$ (framework : 45 000 €/an) et le
coût de migration $\kappa$ ne s'observent pas directement. Décision (06/10/2026) : ils sont
**estimés sur les flux migratoires** (bloc D13) par méthode des moments simulés, pas posés
en paramètres libres. Design et caveats dans `docs/internal/research/estimation-design.md`.
Repli documenté si l'identification échoue : les assumer en structurels, flux en validation.

### D13 — Flux migratoires : identification de A et κ, puis validation

Bloc promu en **identification** : $A$ et $\kappa$ sont estimés sur ces moments par méthode
des moments simulés (design : `docs/internal/research/estimation-design.md`), les mêmes flux
servant ensuite à valider le pic de départ par âge et la hiérarchie des destinations.

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Stock de Français inscrits par pays et par âge/genre | — | pays × âge × genre, annuel | Registre des Français établis hors de France (MEAE), rapport annuel + open data data.gouv.fr | INSEE | web ; CSV open data | manquant |
| Stocks bilatéraux origine→destination par âge/éducation | — | paire × âge × éducation | OCDE *DIOC* version consolidée 2020/21 (détail pays d'origine, sortie 2024) | national | SDMX ; fichiers | manquant |
| Croisement par les registres d'accueil | — | destination × âge | stats nationales d'accueil (CH : OFS/SEM ; US : ACS ; UK : ONS ; etc.) | Eurostat *migr* | portails ; SDMX | manquant |
| Taux de migration de retour | — | destination | OCDE *International Migration Database* | national | SDMX | manquant |
| Taux d'expatriation global (éligibles) | — | national | MEAE stock / population | INSEE | portail | manquant |

### D14 — Démographie pour le rendement de la répartition

| Donnée | Symbole | Granularité | Source primaire | Source secondaire / repli | Accès | Statut |
|--------|---------|-------------|-----------------|---------------------------|-------|--------|
| Croissance démographique | — | pays | OCDE ; ONU *WPP* | national | SDMX | manquant |
| Croissance de la productivité | — | pays | OCDE *Productivity* | national | SDMX | manquant |
| Croissance de la masse salariale | — | pays | dérivé démographie + productivité | projections pension | calculé | manquant |

## Traçabilité

Chaque valeur collectée est consignée avec : millésime, date d'extraction, URL ou
référence exacte, et unité. Le manifeste `sources.yaml` centralise les points
d'accès. Aucune valeur n'est saisie en dur dans le code : les paramètres vivent dans
`data/` et sont lus par `src/data/`.
