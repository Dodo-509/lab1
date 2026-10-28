# Lab 1 – Deep Learning avec PyTorch (Régression & Classification multi-classe)

**Master MBD – FST Tanger** | Pr. ELAACHAK Lotfi

## Objectif
Prendre en main PyTorch pour résoudre une tâche de **régression** et une tâche de **classification multi-classe** avec des architectures MLP/DNN, en passant par l'EDA, la recherche d'hyper-paramètres (`GridSearchCV`) et la régularisation.

## Contenu du dépôt
| Fichier | Description |
|---|---|
| `Lab1_Partie1_Regression_NYSE.ipynb` | Régression sur le dataset NYSE (prédiction du prix de clôture) |
| `Lab1_Partie2_Classification_PyTorch.ipynb` | Classification du type de panne (Predictive Maintenance) |

---

## Partie 1 – Régression (NYSE)

**Démarche** : EDA → split chronologique et normalisation ajustée sur le train → MLP PyTorch → GridSearch (via skorch) → courbes Loss / Accuracy ±2 % / R² → régularisation.

**Résultats de la comparaison des régularisations (test)**

| Modèle | RMSE test (USD) | Accuracy ±2 % | Écart RMSE test–train |
|---|---|---|---|
| Early stopping | ~1,60 | ~0,82 | ~0,5 |
| Base / L1 (1e-5) / L2 (1e-3) | ~1,65 | ~0,81 | ~0,5 |
| Dropout + L2 + ES | ~1,8 | ~0,72 | ~0,5 |
| BatchNorm | ~2,6 | ~0,55 | ~1,0 |
| Dropout 0.2 | ~5,2 | ~0,33 | ~2,5 |
| *Baseline naïve (prix de la veille)* | *~1,55* | | |

**Interprétation**
- Les loss train et test se stabilisent vite et l'écart reste constant : pas de sur-apprentissage classique, l'écart vient du décalage temporel entre train et test.
- Le R² (~0,999) est peu informatif : le prix du jour explique presque tout le prix du lendemain.
- **Aucun modèle ne bat la baseline naïve** : le réseau apprend surtout à recopier le prix du jour.
- L1 et L2 (coefficients faibles) sont équivalents au modèle de base ; le dropout et BatchNorm **dégradent** le modèle (bruit injecté dans la sortie, statistiques de batch non stationnaires).
- Configuration retenue : **Early stopping** (meilleur RMSE et accuracy, nombre d'epochs choisi automatiquement), même si l'écart avec `Base` est dans le bruit (une seule seed).

---

## Partie 2 – Classification multi-classe (Predictive Maintenance)

**Démarche**
1. **Prétraitement** : suppression des identifiants et de `Target` (fuite d'information), retrait des lignes d'étiquetage contradictoire, encodage, standardisation sur le train uniquement.
2. **EDA** : classes très déséquilibrées (~97 % de « No Failure »), corrélation forte `rot_speed` / `torque`, régimes distincts selon le type de panne.
3. **Data augmentation** : **SMOTE-NC** appliqué *après* le split et uniquement sur le train.
4. **Modèle** : MLP PyTorch (`Linear → [BatchNorm] → ReLU → [Dropout]`), encapsulé dans un estimateur compatible sklearn.
5. **GridSearchCV** : `Pipeline` SMOTE-NC → StandardScaler → réseau (l'augmentation est faite dans chaque fold, sans fuite), score = F1 macro. Paramètres : learning rate, optimiseur, epochs, architecture.
6. **Courbes** Loss / Epochs et Accuracy / Epochs (train et test).
7. **Métriques** : accuracy, précision, sensibilité, F1 (macro et pondéré), matrices de confusion.
8. **Régularisation** : Dropout, L2, BatchNorm, Early stopping et combinaison, comparés au modèle de base.

**Résultats** *(à compléter après exécution du notebook)*

| Modèle | Acc test | Sensibilité test (macro) | F1 test (macro) | Écart F1 train–test |
|---|---|---|---|---|
| Base | | | | |
| Dropout | | | | |
| L2 | | | | |
| BatchNorm | | | | |
| Early stopping | | | | |
| Combinaison | | | | |

**Meilleurs hyper-paramètres (GridSearch)** : *à compléter*

---

## Ce que nous avons appris
- **PyTorch** : `nn.Module`, `DataLoader`, boucle d'entraînement (forward / loss / backward / step), optimiseurs, `state_dict`, et intégration à sklearn (`GridSearchCV`).
- **Rigueur méthodologique** : split avant tout traitement, normalisation ajustée sur le train seul, split chronologique pour les séries temporelles, augmentation uniquement sur le train et à l'intérieur des folds de validation croisée.
- **Choisir les bonnes métriques** : l'accuracy est trompeuse sur des classes déséquilibrées (on utilise F1 macro et sensibilité par classe) ; le R² est trompeur sur des séries de prix (on ajoute une accuracy ±2 % et une baseline naïve).
- **Toujours comparer à une baseline** : de bons scores ne prouvent pas qu'un modèle est utile.
- **La régularisation n'est pas magique** : elle n'aide que s'il y a sur-apprentissage. En régression sur données non stationnaires, dropout et BatchNorm peuvent nuire.
- **Importance de l'interprétation des courbes** : l'écart entre loss train et test, et son évolution, permet de distinguer sur-apprentissage et simple décalage de distribution.

## Limites
- Prédire des prix boursiers est un problème très bruité ; résultats basés sur une seule seed et une seule période de test.
- En classification, peu d'exemples réels des classes rares : les données synthétiques (SMOTE) n'apportent pas d'information nouvelle et les métriques de test restent bruitées.
- Pistes : répéter les expériences sur plusieurs seeds (moyenne ± écart-type), validation croisée temporelle, tester d'autres architectures (LSTM pour la partie 1).

## Outils
Python, PyTorch, scikit-learn, imbalanced-learn, skorch, pandas, seaborn / matplotlib – Google Colab / Kaggle, GitHub.
