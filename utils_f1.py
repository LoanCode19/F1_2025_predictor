import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

BAREME_F1 = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
             6: 8,  7: 6,  8: 4,  9: 2, 10: 1}

def pos_to_points(pos: int) -> int:
    return BAREME_F1.get(int(pos), 0)

FEATURES = [
    # Pilote — normalisées
    'forme_recent_norm', 'performance_norm', 'pref_circuit_norm',
    # Pilote — brutes
    'forme_brute', 'performance_brute', 'pref_circuit_brute',
    # Pilote — récence fine & rookie
    'experience', 'is_rookie', 'best_recent_pts', 'avg_recent_pts',
    # Pluie
    'is_wet', 'wet_avg_pts', 'wet_race_count',
    # Équipe — normalisées
    'forme_recent_norm_equipe', 'performance_norm_equipe', 'preference_circuit_norm_equipe',
    # Équipe — brutes
    'forme_brute_equipe', 'performance_brute_equipe', 'pref_circuit_brute_equipe',
    # Contexte
    'round', 'annee',
]

TARGET = 'position_reelle'

def load_and_prepare(path='../Data/merged_features_RF.csv'):
    df = pd.read_csv(path)
    features = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=features + [TARGET, 'pts_sur_le_weekend'])
    return df, features


def train_test_split_temporal(df, n_last=5):
    df_2025        = df[df['annee'] == 2025]
    max_round      = df_2025['round'].max()
    cutoff         = max_round - (n_last - 1)
    mask_test      = (df['annee'] == 2025) & (df['round'] >= cutoff)
    mask_train     = ~mask_test
    print(f"\nSaison 2025 → dernier round : {max_round}")
    print(f"Train : rounds 1 → {cutoff-1} | Test : rounds {cutoff} → {max_round}")
    df_train = df[mask_train].copy()
    df_test  = df[mask_test].copy()
    print(f"Train : {len(df_train)} lignes | Test : {len(df_test)} lignes")
    return df_train, df_test


def make_sample_weights(y_train, w_in_points=2.0, w_out=1.0):
    """Positions 1-10 ont plus de poids."""
    return np.where(y_train <= 10, w_in_points, w_out)


def evaluate(model, X_train, y_train, X_test, y_test, df_test, model_name):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    mae_train      = mean_absolute_error(y_train, y_pred_train)
    mae_test       = mean_absolute_error(y_test,  y_pred_test)
    r2_train       = r2_score(y_train, y_pred_train)
    r2_test        = r2_score(y_test,  y_pred_test)

    mask_pts_train = y_train <= 10
    mask_pts_test  = y_test  <= 10
    mae_top10_train = mean_absolute_error(y_train[mask_pts_train], y_pred_train[mask_pts_train])
    mae_top10_test  = mean_absolute_error(y_test[mask_pts_test],   y_pred_test[mask_pts_test])

    df_test = df_test.copy()
    df_test['pos_predite'] = y_pred_test

    top3_acc  = top3_accuracy(df_test,  'pos_predite', TARGET)
    top10_acc = top10_accuracy(df_test, 'pos_predite', TARGET)

    print(f"\n{'='*55}")
    print(f"  {model_name} — TARGET : position réelle (1 = vainqueur)")
    print(f"  MAE  Train : {mae_train:.3f}  |  Test : {mae_test:.3f}  (toutes positions)")
    print(f"  MAE  Train : {mae_top10_train:.3f}  |  Test : {mae_top10_test:.3f}  (positions 1-10)")
    print(f"  R²   Train : {r2_train:.3f}  |  Test : {r2_test:.3f}")
    print(f"  Précision Top-3  (podium)   : {top3_acc:.1%}")
    print(f"  Précision Top-10 (points)   : {top10_acc:.1%}")
    print(f"{'='*55}")

    return y_pred_test, df_test


def top3_accuracy(df_sub, pred_col, tgt):
    correct, total = 0, 0
    for _, grp in df_sub.groupby('round'):
        real = set(grp.nsmallest(3, tgt)['pilote'])
        pred = set(grp.nsmallest(3, pred_col)['pilote'])
        correct += len(real & pred)
        total   += 3
    return correct / total if total else 0

def top10_accuracy(df_sub, pred_col, tgt):
    correct, total = 0, 0
    for _, grp in df_sub.groupby('round'):
        real = set(grp.nsmallest(10, tgt)['pilote'])
        pred = set(grp.nsmallest(10, pred_col)['pilote'])
        correct += len(real & pred)
        total   += 10
    return correct / total if total else 0


def simulate_races(df_test):
    circuits = df_test.groupby('round')['circuit'].first().sort_index()
    print(f"\n\nSIMULATION DES {len(circuits)} DERNIÈRES COURSES 2025")
    print("=" * 65)
    results = []
    for rnd, circuit_name in circuits.items():
        subset = df_test[df_test['round'] == rnd].copy()
        subset_sorted = subset.sort_values('pos_predite').reset_index(drop=True)
        print(f"\nRound {rnd} — {circuit_name}")
        print(f"  {'Pos prédite':<12} {'Pilote':<25} {'Pts prédits':>11} {'Pos réelle':>10} {'Pts réels':>10}")
        print(f"  {'-'*72}")
        for rank, row in subset_sorted.iterrows():
            pos_p = rank + 1
            pts_p = pos_to_points(pos_p)
            pts_r = row['pts_sur_le_weekend']
            pos_r = int(row[TARGET])
            print(f"  {pos_p:<12} {row['pilote']:<25} {pts_p:>11} {pos_r:>10} {pts_r:>10.0f}")
            results.append({
                'round': rnd, 'circuit': circuit_name,
                'pos_predite': pos_p, 'pos_reelle': pos_r,
                'pilote': row['pilote'], 'team': row['team'],
                'pts_predits': pts_p, 'pts_reels': pts_r,
            })
    return pd.DataFrame(results)


def print_cumul(df_results):
    print(f"\n\nCLASSEMENT CUMULÉ SUR LES {len(df_results['round'].unique())} COURSES SIMULÉES")
    print("=" * 55)
    cp = df_results.groupby('pilote')['pts_predits'].sum().sort_values(ascending=False)
    cr = df_results.groupby('pilote')['pts_reels'].sum()
    cumul = pd.DataFrame({'pts_predits': cp, 'pts_reels': cr}).fillna(0)
    cumul['delta'] = cumul['pts_predits'] - cumul['pts_reels']
    print(cumul.to_string())
    return cumul


def plot_results(model, features, y_test, y_pred_test, cumul, model_name, save_path):
    feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n\nImportance des features ({model_name}) :")
    print(feat_imp.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(model_name, fontsize=13, fontweight='bold')

    feat_imp.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].invert_yaxis()
    axes[0].set_title("Importance des features")
    axes[0].set_xlabel("Importance")

    axes[1].scatter(y_test, y_pred_test, alpha=0.4, color='darkorange', edgecolors='k', linewidths=0.3)
    axes[1].plot([1, 20], [1, 20], 'r--', lw=1.5, label='Parfait')
    axes[1].set_xlabel("Position réelle")
    axes[1].set_ylabel("Position prédite")
    mae = mean_absolute_error(y_test, y_pred_test)
    r2  = r2_score(y_test, y_pred_test)
    axes[1].set_title(f"Prédit vs Réel\nMAE={mae:.2f} pos  R²={r2:.2f}")
    axes[1].legend()

    top15 = cumul.sort_values('pts_reels', ascending=False).head(15)
    x = np.arange(len(top15))
    w = 0.35
    axes[2].bar(x - w/2, top15['pts_predits'], w, label='Prédits', color='steelblue')
    axes[2].bar(x + w/2, top15['pts_reels'],   w, label='Réels',   color='coral')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(top15.index, rotation=45, ha='right', fontsize=8)
    axes[2].set_title(f"Pts cumulés — Top 15 ({len(cumul)} courses)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Graphique sauvegardé : {save_path}")
