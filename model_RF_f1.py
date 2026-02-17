from sklearn.ensemble import RandomForestRegressor
from utils_f1 import (load_and_prepare, train_test_split_temporal,
                      make_sample_weights, evaluate,
                      simulate_races, print_cumul, plot_results, TARGET)

def predict_and_evaluate(nth_last_races=5):
    df, FEATURES = load_and_prepare('Data/merged_features_RF.csv')
    df_train, df_test = train_test_split_temporal(df, n_last=nth_last_races)
    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]
    sample_weights = make_sample_weights(y_train) # Positions 1-10 ont plus de poids

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=8,
        min_samples_split=15,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Random Forest entraîné")
    _, df_test_pred = evaluate(model, X_train, y_train, X_test, y_test, df_test, "Random Forest")

    df_results = simulate_races(df_test_pred)
    cumul      = print_cumul(df_results)

    plot_results(model, FEATURES, y_test, model.predict(X_test),
                cumul, "Random Forest F1", f"Data/rf_evaluation_{nth_last_races}_last_races.png")

    df_results.to_csv(f'Data/simulation_{nth_last_races}_last_races_2025_RF.csv', index=False)
    print(f"Résultats sauvegardés : Data/simulation_{nth_last_races}_last_races_2025_RF.csv")
    return
