import fastf1
import pandas as pd
import numpy as np


def curr_team_performance(dico_recence, liste_teams, window=5, base=1.3):
    """Forme récente pondérée exponentiellement (plus récent = plus de poids)."""
    dico_forme = {}
    for team in liste_teams:
        recent = dico_recence[team][-window:]
        if len(recent) == 0:
            dico_forme[team] = 0.0
            continue
        exps    = np.arange(len(recent))
        weights = base ** exps
        dico_forme[team] = float(np.dot(recent, weights))
    return dico_forme


def pref_circuit_team(dico_res_team_circuit, liste_teams, circuit):
    """Préférence (pondérée chronologiquement) d'une équipe pour un circuit."""
    dico_pref_retour = {team: 0.0 for team in liste_teams}
    for team in liste_teams:
        liste_pref = dico_res_team_circuit[team].get(circuit, [])
        if not liste_pref:
            dico_pref_retour[team] = 0.0
        else:
            for i, v in enumerate(liste_pref):
                dico_pref_retour[team] += v * (i + 1)
    return dico_pref_retour


# ── Initialisation à partir du CSV base ────────────────────────────────────────
df_teams = pd.read_csv('Data/moyenne_teams2022.csv')

dico_performance       = {t: [pts * 0.05] for t, pts in zip(df_teams['team'], df_teams['points_total_coeffed'])}
dico_recence           = {t: []           for t in df_teams['team']}
dico_res_team_circuit  = {t: {}           for t in df_teams['team']}
liste_teams            = list(dico_recence.keys())

fastf1.Cache.enable_cache('cache_folder')

rows = []

for season in range(2022, 2026):
    i = 0
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    for _, event in schedule.iterrows():
        nom = event.EventName
        if not event.is_testing():
            race = event.get_race()
            race.load(laps=False, telemetry=False, weather=False, messages=False)
            if race.results.empty:
                continue

            team_points = race.results.groupby('TeamId')['Points'].sum().to_dict()

            # ── Features brutes ────────────────────────────────────────────
            dico_forme = curr_team_performance(dico_recence, liste_teams, window=5, base=1.3)
            dico_pref  = pref_circuit_team(dico_res_team_circuit, liste_teams, nom)

            forme_vals = np.array([dico_forme[t]              for t in liste_teams])
            perf_vals  = np.array([dico_performance[t][-1]    for t in liste_teams])
            pref_vals  = np.array([dico_pref[t]               for t in liste_teams])

            # ── Normalisation ──────────────────────────────────────────────
            forme_norm = (forme_vals - forme_vals.min()) / (forme_vals.max() - forme_vals.min() + 1e-9)
            perf_norm  = (perf_vals  - perf_vals.min())  / (perf_vals.max()  - perf_vals.min()  + 1e-9)
            pref_norm  = (pref_vals  - pref_vals.min())  / (pref_vals.max()  - pref_vals.min()  + 1e-9)

            for idx, team in enumerate(liste_teams):
                rows.append({
                    'team'                          : team,
                    'round'                         : i + 1,
                    'annee'                         : season,
                    # ── normalisées ───────────────────────────────────────
                    'forme_recent_norm_equipe'      : forme_norm[idx],
                    'performance_norm_equipe'       : perf_norm[idx],
                    'preference_circuit_norm_equipe': pref_norm[idx],
                    # ── brutes ────────────────────────────────────────────
                    'forme_brute_equipe'            : dico_forme[team],
                    'performance_brute_equipe'      : dico_performance[team][-1],
                    'pref_circuit_brute_equipe'     : dico_pref[team],
                })

            for team in liste_teams:
                pts = team_points.get(team, 0.0)
                dico_recence[team].append(pts)
                dico_performance[team].append(dico_performance[team][-1] + pts)
                dico_res_team_circuit[team].setdefault(nom, []).append(pts)

            i += 1

df_final = pd.DataFrame(rows)
df_final.to_csv('Data/teams_features_RF.csv', index=False)
print("CSV 'teams_features_RF.csv' créé !")
