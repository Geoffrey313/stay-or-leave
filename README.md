# Rester ou partir : un modèle dynamique d'arbitrage fiscal international du pouvoir d'achat sur le cycle de vie

Auteur : Geoffrey Ducournau.

## Résumé

La décision d'un salarié qualifié de rester dans son pays ou d'émigrer est formalisée
comme un arbitrage de valeur actualisée patrimoniale sur le cycle de vie, sous coût de
migration et sous incertitude, résolu par une équation de Bellman en récursion arrière. Le
modèle assemble, pour un agent français face
à huit destinations, une cascade socio-fiscale microsimulée du coût employeur au pouvoir
d'achat, un coût de la vie, un flux immobilier, un surplus d'épargne investi, des
contrepoids socialisés et une prime salariale à rôle comparable, puis estime l'attachement
au pays et le coût de migration sur les flux migratoires observés. Résultats principaux :
la France convertit le coût du travail en pouvoir d'achat de la façon la plus régressive
des pays finement mesurés, mais à rôle comparable et en pouvoir d'achat réel le cadre
médian ne gagne pas à partir ; le départ est un fait d'occupation d'élite et de
destination ; le moteur de retraite n'opère que vers un pilier capitalisé obligatoire, la
Suisse, tandis que l'avantage américain transite par le surplus d'épargne ; la propriété
du logement retient de premier ordre et encadre l'émigration entre quatre et six pour cent.
Le travail déplace le débat d'une arithmétique de taux nominaux vers une comparaison de
bien-être de cycle de vie, et montre que la régressivité française ne se traduit pas en
fuite généralisée.

## Questions de recherche

1. Où se situe la France dans la conversion du coût du travail en pouvoir d'achat, du bas
   au haut de la distribution des gains, parmi neuf pays ?
2. À partir de quel niveau de revenu, et pour quelle destination, l'émigration maximise la
   valeur patrimoniale de cycle de vie, et comment ce seuil dépend-il de l'âge et de la
   réversibilité ?
3. Quel mécanisme économique porte l'arbitrage, et dépend-il de la destination ?
4. Quels contrepoids socialisés inversent la conclusion au bas de la distribution ?

## Hypothèses

- **H1.** La conversion française du coût du travail en pouvoir d'achat est la plus
  régressive des pays finement mesurés : le taux de conversion décroît avec le revenu et
  atteint son plus bas au sommet.
- **H2.** À rôle comparable et en pouvoir d'achat réel, le cadre médian ne gagne pas à
  émigrer : la prime observée reste en deçà du seuil de bascule, si bien que le départ est
  un fait d'occupation d'élite et de destination.
- **H3.** La bascule de la retraite vers un actif de marché n'opère que vers une
  destination dotée d'un pilier capitalisé obligatoire ; ailleurs, l'avantage éventuel
  transite par le surplus d'épargne investi.
- **H4.** Les contrepoids socialisés inversent le classement au bas de la distribution, et
  la propriété du logement retient de premier ordre.

## Résultats principaux

Chaque chiffre est une sortie de `reproduce.py`.

- **H1.** Le taux de conversion brut du profil célibataire décroît en France de 72,8 % au
  premier décile à 36,2 % au centile le plus élevé, contre 82,2 % à 62,5 % aux États-Unis
  et 74,2 % à 56,7 % en Suisse. La France a la pente la plus négative.
- **H2.** Le seuil de prime à rôle comparable au-delà duquel partir domine monte de 1,10 à
  1,34 vers les États-Unis et de 1,05 à une valeur jamais atteinte vers le Royaume-Uni. La
  prime observée vaut 0,97 vers les États-Unis et 0,79 vers le Royaume-Uni, sous le seuil à
  toute position : le cadre type reste. Au sommet occupationnel, la prime des dirigeants
  vaut 1,27 vers les États-Unis (partir) et 0,85 vers le Royaume-Uni (rester).
- **H3.** À aménité nulle, la part d'agents émigrant vers la Suisse vaut l'unité, contre
  0,07 vers les États-Unis : le pilier capitalisé obligatoire suisse fait opérer la
  bascule, que l'absence de pilier obligatoire américain laisse passer par le surplus.
- **H4.** Le taux de conversion élargi dépasse l'unité au bas de la distribution pour les
  profils avec enfants (couple au premier décile : 1,22 en France, 1,73 aux États-Unis).
  Activer la propriété fait baisser la part nette à l'étranger, et le verrouillage
  universel sur-retient : l'émigration est encadrée entre quatre et six pour cent.

## Reproduction

```
python -m pip install -r requirements.txt
python reproduce.py            # chemin par défaut rapide
python reproduce.py --full     # ajoute la courbe A*(role_ratio), lourde, déterministe
```

La chaîne est déterministe : lecture des paramètres de `data/`, calcul de la cascade et
des flux, résolution du benchmark statique puis de la Bellman dynamique, porte du solveur
et estimation par méthode des moments simulés, production des tableaux et figures. Les
données d'entrée sont des paramètres millésime 2026, documentés et tracés dans
`data/README.md`. Le chemin par défaut estime l'aménité et le coût de migration au seul
`role_ratio` sourcé ; la courbe `A*(role_ratio)`, qui est le résultat de publication (le
seuil, pas un verdict), est régénérée par `python reproduce.py --full`. Les deux chemins
passent par la même porte d'acceptation du solveur.

## Structure

```
src/common/     constantes, paramètres de protocole, chemins portables
src/data/       lecture des paramètres transformés livrés dans data/
src/engine/     cascade fiscale, flux patrimonial, benchmark statique, Bellman
src/analysis/   seuils, décompositions, estimation et validation sur flux
src/figures/    scripts des figures et tableaux générés
data/           paramètres d'entrée versionnés + registre des sources
reproduce.py    point d'entrée déterministe de bout en bout
```

Les manuscrits (`manuscript/`, versions française et anglaise) et les notes de travail
(`docs/internal/`) vivent dans l'espace de travail mais ne sont pas publiés.

## Contact

G.ducournau.voisin@gmail.com
