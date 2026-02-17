import model_RF_f1


print("Rentrez le nombre de course qui seront simulées (ex: 3, Vegas,Qatar et Abou Dabi) :")
while True:
    try:
        nth_last = int(input())
        if 1 <= nth_last <= 23:
             break
        else:
            print("Veuillez entrer un nombre entre 1 et 23 :")
    except ValueError:
        print("Veuillez entrer un nombre valide.")

model_RF_f1.predict_and_evaluate(nth_last_races=nth_last)