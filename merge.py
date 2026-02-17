import pandas as pd

pilotes = pd.read_csv('Data/pilotes_features_RF.csv', sep=',')
team    = pd.read_csv('Data/teams_features_RF.csv',   sep=',')

# ── Harmonisation des noms d'équipes historiques ──────────────────────────
for df in [pilotes, team]:
    df['team'] = df['team'].replace({'alphatauri': 'rb', 'alfa': 'sauber'})

merged_df = pd.merge(
    pilotes,
    team,
    on=['team', 'round', 'annee'],
    how='left',
)

merged_df.to_csv('Data/merged_features_RF.csv', sep=',', index=False)
print("Données fusionnées et sauvegardées : Data/merged_features_RF.csv")
