import fastf1
import pandas as pd
import numpy as np

def poids_recence(annee):
    """Poids entre 0.05 (en 2010) et ~1 (en 2022)."""
    return 0.05 * np.exp(0.25 * (annee - 2010))

def form_teams():
    """Récupère la liste des équipes de la saison 2025 et renvoie le dico de performance des équipes."""
    fastf1.Cache.enable_cache('cache_folder', use_requests_cache=True)
    session = fastf1.get_session(2025, 'United States', 'R')
    session.load()
    liste_teams = session.results["TeamId"].unique().tolist()
    points_total_team_coeffed = {team: 0 for team in liste_teams}
    anciennete_team = {team: 0 for team in liste_teams}
    for season in range(2010,2022):
        schedule = fastf1.get_event_schedule(season)
        print(f'\n\n\n\n---------------------------------------------------- Nouvelle saison : {season} ----------------------------------------------------\n\n\n\n\n\n\n')
        for _, event in schedule.iterrows():
            if not event.is_testing():
                race = event.get_race()
                race.load(laps=False, telemetry=False, weather=False, messages=False)
                for point,team in zip(race.results['Points'],race.results['TeamId']):
                    if team in liste_teams:
                        points_total_team_coeffed[team] += point * poids_recence(season)
                        anciennete_team[team] += 1/2
                    elif team == 'alphatauri' or team == 'toro_rosso':
                        points_total_team_coeffed['rb'] += point * poids_recence(season)
                        anciennete_team['rb'] += 1/2
                    elif team == 'lotus_f1' or team == 'renault':
                        points_total_team_coeffed['alpine'] += point * poids_recence(season)
                        anciennete_team['alpine'] += 1/2
                    elif team == 'racing_point' or team == 'force_india':
                        points_total_team_coeffed['aston_martin'] += point * poids_recence(season)
                        anciennete_team['aston_martin'] += 1/2
                    elif team == 'alfa' or team == 'bmw_sauber':
                        points_total_team_coeffed['sauber'] += point * poids_recence(season)
                        anciennete_team['sauber'] += 1/2
    return points_total_team_coeffed, anciennete_team

points_total_team_coeffed, anciennete_team = form_teams()
print("Performance des équipes (points totaux coeffés) :")
for team in points_total_team_coeffed:
    print(f"{team} : {points_total_team_coeffed[team]:.2f} points sur {anciennete_team[team]} courses, moyenne coeffée : {points_total_team_coeffed[team]/anciennete_team[team]:.2f}")

df_teams = pd.DataFrame({
    'team': list(points_total_team_coeffed.keys()),
    'points_total_coeffed': list(points_total_team_coeffed.values()),
    'anciennete': [anciennete_team[team] for team in points_total_team_coeffed.keys()]
})

df_teams['moyenne_coeffee'] = df_teams['points_total_coeffed'] / df_teams['anciennete']

df_teams.to_csv('Data/moyenne_teams2022.csv', index=False)
print("CSV 'moyenne_teams2022.csv' créé avec succès !")
print(df_teams[['team','moyenne_coeffee']])


                


