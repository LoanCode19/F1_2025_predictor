import fastf1
import numpy as np
import pandas as pd

def calculate_form(dico_exp, dico_resultats, liste_pilotes, window_size=5):
    dico_forme_retour = {}
    for pilote in liste_pilotes:
        if dico_exp[pilote][-1] < 1:
            dico_forme_retour[pilote] = 0.0
            continue
        if len(dico_resultats[pilote]) < window_size + 1:
            window = dico_resultats[pilote][:-1]
        else:
            window = dico_resultats[pilote][-window_size-1:-1]
        form = 0
        for i in range(len(window)):
            form += window[i] * (1.3 ** i)
        dico_forme_retour[pilote] = form
    return dico_forme_retour


def pond(liste_pref_course):
    if len(liste_pref_course) == 0:
        return 0.0
    total = 0.0
    for i in range(len(liste_pref_course)):
        total += liste_pref_course[i] * (i + 1)
    return total


def is_race_wet(race):
    """
    Retourne 1 si la course s'est déroulée sous la pluie, 0 sinon.
    Stratégie : on charge les données météo et on regarde si
    Rainfall == True sur plus de 20% des lignes.
    En cas d'erreur on retourne 0 (course sèche par défaut).
    """
    try:
        weather = race.weather_data
        if weather is None or weather.empty:
            return 0
        if 'Rainfall' in weather.columns:
            pct_rain = weather['Rainfall'].sum() / len(weather)
            return 1 if pct_rain > 0.2 else 0
    except Exception:
        pass
    return 0


df_pilotes = pd.read_csv('Data/driver_base_level_from_2010.csv')
dico_experience          = {p: [exp]        for p, exp in zip(df_pilotes['DriverId'], df_pilotes['Experience'])}
dico_level_performance   = {p: [pts * 0.05] for p, pts in zip(df_pilotes['DriverId'], df_pilotes['BaseLevel'])}
dico_team_pilotes        = {p: 0            for p in df_pilotes['DriverId']}
dico_pts_weekend         = {p: []           for p in df_pilotes['DriverId']}
dico_pref_pilote_circuit = {p: {}           for p in df_pilotes['DriverId']}

# ── Suivi des performances sous la pluie par pilote ───────────────────────
# On stocke les pts marqués lors des courses humides pour calculer
# une "forme sous la pluie" (wet_avg_pts)
dico_wet_pts = {p: [] for p in df_pilotes['DriverId']}

fastf1.Cache.enable_cache('cache_folder')
seasons = [2022, 2023, 2024, 2025] # On peut ajuster les saisons à charger (2025 incluse pour les features récentes)
rows = []

for season in seasons:
    i = 0
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    for _, event in schedule.iterrows():
        i += 1
        nom = event.EventName
        liste_pilotes_curr = []
        position_map = {}

        if not event.is_testing():
            race = event.get_race()
            # On charge la météo en plus
            race.load(laps=False, telemetry=False, weather=True, messages=False)

            # ── Détection pluie ───────────────────────────────────────────
            wet = is_race_wet(race)

            for driver, points, team, position in zip(
                race.results['DriverId'],
                race.results['Points'],
                race.results['TeamId'],
                race.results['Position'],
            ):
                if driver not in liste_pilotes_curr:
                    liste_pilotes_curr.append(driver)
                dico_pref_pilote_circuit[driver].setdefault(nom, [])
                dico_team_pilotes[driver] = team
                dico_pts_weekend[driver].append(points)
                try:
                    position_map[driver] = int(position)
                except (ValueError, TypeError):
                    position_map[driver] = 20

        else:
            wet = 0

        if len(liste_pilotes_curr) == 0:
            continue

        # ── Features brutes ────────────────────────────────────────────────
        dico_forme = calculate_form(dico_experience, dico_pts_weekend, liste_pilotes_curr, window_size=5)

        pref_brut = {
            p: pond(dico_pref_pilote_circuit[p].get(nom, []))
            for p in liste_pilotes_curr
        }

        # ── Normalisation (contexte course) ───────────────────────────────
        forme_vals = np.array([dico_forme[p]                  for p in liste_pilotes_curr])
        perf_vals  = np.array([dico_level_performance[p][-1]  for p in liste_pilotes_curr])
        pref_vals  = np.array([pref_brut[p]                   for p in liste_pilotes_curr])

        forme_norm = (forme_vals - forme_vals.min()) / (forme_vals.max() - forme_vals.min() + 1e-9)
        perf_norm  = (perf_vals  - perf_vals.min())  / (perf_vals.max()  - perf_vals.min()  + 1e-9)
        pref_norm  = (pref_vals  - pref_vals.min())  / (pref_vals.max()  - pref_vals.min()  + 1e-9)

        for idx, pilote in enumerate(liste_pilotes_curr):
            exp_curr = dico_experience[pilote][-1]

            # ── Feature rookie ─────────────────────────────────────────────
            is_rookie = 1 if exp_curr < 10 else 0

            # Récence fine (3 dernières courses)
            recent_pts   = dico_pts_weekend[pilote][-3:] if len(dico_pts_weekend[pilote]) >= 3 \
                           else dico_pts_weekend[pilote]
            best_recent_pts = max(recent_pts) if recent_pts else 0.0
            avg_recent_pts  = float(np.mean(recent_pts)) if recent_pts else 0.0

            # ── Feature pluie ──────────────────────────────────────────────
            # wet_avg_pts : moyenne des points marqués par ce pilote
            #               lors des courses humides passées.
            #               Donne la capacité du pilote à performer sous la pluie.
            wet_history = dico_wet_pts[pilote]
            wet_avg_pts = float(np.mean(wet_history)) if wet_history else avg_recent_pts
            # wet_race_count : nb de courses humides disputées (confiance du signal)
            wet_race_count = len(wet_history)

            rows.append({
                'circuit'             : nom,
                'pilote'              : pilote,
                'team'                : dico_team_pilotes[pilote],
                'round'               : i,
                'annee'               : season,
                # ── normalisées ───────────────────────────────────────────
                'forme_recent_norm'   : forme_norm[idx],
                'performance_norm'    : perf_norm[idx],
                'pref_circuit_norm'   : pref_norm[idx],
                # ── brutes ────────────────────────────────────────────────
                'forme_brute'         : dico_forme[pilote],
                'performance_brute'   : dico_level_performance[pilote][-1],
                'pref_circuit_brute'  : pref_brut[pilote],
                # ── rookie & récence fine ─────────────────────────────────
                'experience'          : exp_curr,
                'is_rookie'           : is_rookie,
                'best_recent_pts'     : best_recent_pts,
                'avg_recent_pts'      : avg_recent_pts,
                # ── pluie ─────────────────────────────────────────────────
                'is_wet'              : wet,           # 1 = course sous la pluie
                'wet_avg_pts'         : wet_avg_pts,   # perf historique sous la pluie
                'wet_race_count'      : wet_race_count,# nb courses humides vues
                # ── target ────────────────────────────────────────────────
                'pts_sur_le_weekend'  : dico_pts_weekend[pilote][-1],
                'position_reelle'     : position_map.get(pilote, 20),
            })

            # Mise à jour des dicos APRÈS enregistrement
            dico_experience[pilote].append(exp_curr + 1)
            dico_level_performance[pilote].append(
                dico_level_performance[pilote][-1]
                + dico_pts_weekend[pilote][-1] * (1 + 1 / (exp_curr + 1))
            )
            dico_pref_pilote_circuit[pilote][nom].append(dico_pts_weekend[pilote][-1])
            # Mise à jour historique pluie
            if wet:
                dico_wet_pts[pilote].append(dico_pts_weekend[pilote][-1])

df_final = pd.DataFrame(rows)
df_final.to_csv('Data/pilotes_features_RF.csv', index=False)
print(f"CSV 'pilotes_features_RF.csv' créé ! ({len(df_final)} lignes)")
