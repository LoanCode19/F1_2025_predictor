# F1 Race Predictor

Modèle Random Forest pour prédire les résultats de courses F1 en 2025, entraîné sur les données 2010-2025 via la librairie FastF1.

---

## Structure actuelle

```
.
├── Driver_init_data.py   # Calcule le niveau de base de chaque pilote sur 2010-2021
├── Team_init_data.py     # Idem pour les équipes
├── Data_pilotes.py       # Construit les features pilotes course par course (2022-2025)
├── Data_teams.py         # Idem pour les équipes
├── merge.py              # Fusionne les deux CSV pilotes + équipes
├── model_RF_f1.py        # Entraîne le Random Forest et évalue les prédictions
├── utils_f1.py           # Fonctions partagées (métriques, simulation, plots)
├── main.py               # Point d'entrée
│
├── Data/                 # CSV intermédiaires et résultats (généré automatiquement)
└── cache_folder/         # Cache FastF1 (peut peser plusieurs Go — dans le .gitignore)
```

---

## Pipeline complet

Les scripts sont à exécuter dans cet ordre :

**Étape 1 — Données initiales** (à faire une seule fois, couvre 2010-2021)
```bash
cd F1_Predictior #nécéssaire car les chemins sont spécifiés en relatif
python Driver_init_data.py
python Team_init_data.py
```
Ces scripts appellent l'API FastF1 sur 12 saisons, ça prend du temps. Les CSVs qui seront créées après l'exécution de ces fichiers sont déjà dans le folder Data (driver_base_level_from_2010 et moyenne_teams2022). Ces éxéctutions devront être executées plusieurs fois d'affilée à cause de la limite d'api non gérée.

**Étape 2 — Construction des features** (2022-2025)
```bash
python Data_pilotes.py
python Data_teams.py
python merge.py
```
Les CSVs (respectivement pilotes_features_RF, teams_features_RF et merged_features_RF) se trouvent aussi dans le folder data et l'éxécution des fichiers se heurte au même problème qu'au desssus si les données ne sont pas dans le cache.

**Étape 3 — Entraînement et prédiction**
```bash
python main.py
```
Le script demande combien de dernières courses utiliser pour le test (entre 1 et 23).

---

## Features

**Pilote**
- Forme récente (fenêtre de 5 courses, pondération exponentielle)
- Niveau de performance cumulé depuis 2010
- Affinité historique avec le circuit
- Expérience totale + flag rookie (< 10 courses)
- Récence fine : meilleur et moyen des 3 dernières courses
- Performances sous la pluie (moyenne historique + nombre de courses humides disputées)

**Équipe**
- Forme récente, performance cumulée, affinité circuit (même logique que pilote)

**Contexte**
- Numéro de manche, saison

---

## Sorties

- `Data/merged_features_RF.csv` — dataset final utilisé pour l'entraînement
- `Data/simulation_N_last_races_2025_RF.csv` — résultats de simulation course par course
- `Data/rf_evaluation_N_last_races.png` — graphiques : importance des features, prédit vs réel, points cumulés

---

## Dépendances

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib
```

---

## Notes

- Les changements de noms d'équipes sont normalisés (`alphatauri` → `rb`, `alfa` → `sauber`)
- Target : position finale (1 = vainqueur), le modèle prédit un score continu qu'on rerange ensuite
