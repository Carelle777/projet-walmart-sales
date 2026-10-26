# Walmart Sales Prediction 🛒

## Contexte du Projet
Le service marketing de Walmart souhaite mieux comprendre l'influence des indicateurs économiques (température, prix du carburant, taux de chômage, etc.) sur les ventes hebdomadaires de ses magasins. 
L'objectif de ce projet est de construire un modèle de **Machine Learning supervisé (Régression)** capable d'estimer ces ventes hebdomadaires avec la meilleure précision possible, afin d'aider à la planification des futures campagnes marketing.

## Objectifs
Ce projet valide les compétences d'analyse prédictive sur données structurées, divisé en trois parties :
1. **Analyse Exploratoire (EDA) & Preprocessing :** Nettoyage des données, gestion des valeurs manquantes, feature engineering temporel (extraction des dates) et exclusion des outliers (règle des 3 écarts-types).
2. **Modèle de Référence (Baseline) :** Entraînement d'une Régression Linéaire classique.
3. **Lutte contre le Surapprentissage (Overfitting) :** Entraînement d'un modèle régularisé (Ridge), avec optimisation des hyperparamètres via `GridSearchCV` et évaluation par validation croisée (K-Fold).

## Stack Technique
* **Langage :** Python
* **Manipulation de données :** Pandas, Numpy
* **Machine Learning :** Scikit-Learn (Pipeline, ColumnTransformer, LinearRegression, Ridge, GridSearchCV)
* **Visualisation :** Matplotlib, Seaborn

## Pipeline de Traitement (Scikit-Learn)
Pour garantir l'absence de fuite de données (Data Leakage), le preprocessing est encapsulé :
* **Variables numériques :** Imputation par la moyenne & Standardisation (`StandardScaler`).
* **Variables catégorielles :** Imputation par la valeur la plus fréquente & Encodage binaire (`OneHotEncoder` avec `drop='first'` et `handle_unknown='ignore'`).

## Résultats
Après nettoyage (valeurs manquantes sur la cible et outliers retirés), il reste **131 lignes**.

| Modèle | R² entraînement | R² test |
|---|---|---|
| Régression linéaire (baseline) | 0,973 | 0,934 |
| Ridge optimisé (alpha = 0,01) | 0,973 | 0,936 |

* La validation croisée à 3 plis avec Ridge donne un R² de 0,819 avant réglage, et de 0,945 avec le meilleur alpha trouvé par `GridSearchCV`.
* En mots simples : le modèle explique environ **94 %** des variations des ventes sur des données qu'il n'a jamais vues.

## Variables importantes (coefficients du modèle Ridge)
* **Le magasin est la variable la plus importante.** Les 10 plus gros coefficients sont tous des magasins (par rapport au magasin 1, pris comme référence). Exemple : le magasin 5 vend environ 1,4 million de moins par semaine, les magasins 14 et 10 environ 0,7 million de plus.
* **Les variables économiques pèsent beaucoup moins** (variables standardisées) : chômage -93 000, CPI +78 000, mois +65 000, température -38 000, prix du carburant -35 000.
* La colonne `day_of_week` a un coefficient nul : elle ne varie pas dans les données.

## Limites
* Petit jeu de données (131 lignes) et un seul découpage 80/20 : les résultats peuvent varier avec un autre découpage.
* L'alpha retenu est très faible (0,01) : Ridge est presque identique à la régression linéaire, la régularisation apporte seulement un petit gain.
* Les coefficients montrent des liens statistiques, pas des causes.
