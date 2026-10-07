# Rester ou partir : un modèle dynamique d'arbitrage fiscal international du pouvoir d'achat sur le cycle de vie

Auteurs : Geoffrey Ducournau. [TODO : co-auteurs à confirmer.]

## Résumé

La décision d'un salarié de rester dans son pays ou d'émigrer est formalisée comme un
arbitrage de valeur actualisée patrimoniale sur le cycle de vie, sous coût de migration.
Le cœur du modèle est une bascule de la composante retraite : en répartition elle rend
un rendement faible, proche de la croissance de la masse salariale ; en capitalisation
elle devient un actif composé au rendement de marché. Le modèle assemble, pour neuf pays,
une cascade socio-fiscale coût employeur vers pouvoir d'achat, un coût de la vie, un flux
immobilier, un surplus d'épargne, cinq contrepoids socialisés et une prime salariale,
puis les résout dans une équation de Bellman par récursion arrière.

## Questions de recherche

1. Où se situe la France dans la conversion du coût du travail en pouvoir d'achat, du bas
   au haut de la distribution, parmi neuf pays ?
2. À partir de quel niveau de revenu l'émigration maximise la valeur patrimoniale de cycle
   de vie, et comment ce seuil dépend-il de l'âge, de la destination et de la réversibilité ?
3. Quelle est la structure complète des flux origine vers destination entre les neuf pays ?
4. Quels contrepoids socialisés inversent la thèse pour les bas revenus et l'érodent avec
   l'âge et la parentalité ?

## Hypothèses

[TODO : formaliser H1..Hk à partir des questions, chacune ancrée dans la littérature,
mécanisme et signe attendu posés, conformément à ssrn-working-paper §3.]

## Résultats principaux

[TODO : à renseigner une fois la chaîne de reproduction exécutée. Chaque chiffre ici sera
une sortie de `reproduce.py`, jamais une valeur saisie à la main.]

## Reproduction

```
python -m pip install -r requirements.txt
python reproduce.py            # chemin par défaut rapide
python reproduce.py --full     # ajoute la courbe A*(role_ratio), lourde, déterministe
```

La chaîne est déterministe : lecture des paramètres de `data/`, calcul de la cascade et
des flux, résolution du benchmark statique puis de la Bellman dynamique, porte du solveur
et estimation SMM, production des tableaux et figures. Les données d'entrée sont des
paramètres millésime 2026, documentés et tracés dans `data/README.md`.

Le chemin par défaut estime l'aménité et le coût de migration (SMM, bloc D13) au seul
`role_ratio` sourcé (proxy éducation) : rapide. La **courbe `A*(role_ratio)`**, qui est le
résultat de publication (le seuil, pas un verdict), est plus lourde (une SMM complète par
ancre de `role_ratio`) ; elle est régénérée de façon déterministe par `python reproduce.py
--full`. Les deux chemins passent par la même porte d'acceptation du solveur. Les sources,
leur granularité et leur statut de collecte figurent dans la table de `data/README.md`.

## Structure

```
src/common/     constantes, paramètres de protocole, chemins portables
src/data/       lecture des paramètres transformés livrés dans data/
src/engine/     cascade fiscale, flux patrimonial, benchmark statique, Bellman
src/analysis/   seuils, décompositions, matrice de flux, validation
src/figures/    scripts des figures et tableaux générés (EN et FR)
data/           paramètres d'entrée versionnés + registre des sources
reproduce.py    point d'entrée déterministe de bout en bout
```

## Contact

G.ducournau.voisin@gmail.com
