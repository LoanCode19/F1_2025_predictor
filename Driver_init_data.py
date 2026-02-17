import fastf1
import numpy as np
import pandas as pd
import math

def poids_recence(annee, annee_min=2010, annee_max=2022, poids_min=0.3, poids_max=1.0):
    """
    Renvoie un poids exponentiel pour une année donnée.
    Plus l'année est récente, plus le poids est proche de poids_max.
    """
    pente = 0.7/11
    poids = poids_min + pente * (annee - annee_min)
    return poids
            
def driver_liste():
    fastf1.Cache.enable_cache('cache_folder', use_requests_cache=True)
    session = fastf1.get_session(2025, 'United States', 'R')
    session.load()
    tlpilotes = set()
    for annee in range(2022, 2026):  # inclut 2025
        saison = fastf1.get_event_schedule(annee)
        for _, event in saison.iterrows():
            if not event.is_testing():
                session = fastf1.get_session(annee, event['EventName'], 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                pilotes = session.results["DriverId"].tolist()
                tlpilotes.update(pilotes)
    df_pilotes = pd.DataFrame(sorted(list(tlpilotes)), columns=["DriverId"])
    df_pilotes.to_csv('Data/driver_list_from_2022.csv', index=False)

def driver_base_level():
    fastf1.Cache.enable_cache('cache_folder', use_requests_cache=True)
    df = pd.read_csv("Data/driver_list_from_2022.csv")
    drivers = df["DriverId"].tolist()
    driver_level = {driver: 0 for driver in drivers}
    driver_experience = {driver: 0 for driver in drivers}
    for season in range(2010, 2022):
        schedule = fastf1.get_event_schedule(season)
        for _, event in schedule.iterrows():
            if not event.is_testing():
                race = event.get_race()
                race.load(laps=False, telemetry=False, weather=False, messages=False)
                for point,pilote in zip(race.results['Points'],race.results['DriverId']):
                    if pilote in drivers:
                        poids = poids_recence(season)
                        driver_level[pilote] += point * poids
                        driver_experience[pilote] += 1
    data_frame = pd.DataFrame({
        'DriverId': drivers,
        'BaseLevel': [driver_level[driver] for driver in drivers],
        'Experience': [driver_experience[driver] for driver in drivers]
    })
    data_frame.to_csv('Data/driver_base_level_from_2010.csv', index=False)
    return driver_level, driver_experience

print(driver_base_level())