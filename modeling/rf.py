import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def train_rf_with_optuna(X_train, y_train, X_test, y_test, best_params):
    """
    Train a RandomForest using Optuna best_params.
    """

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    nrmse = rmse / np.mean(y_test)

    return mae, rmse, nrmse


def objective_for_rf(trial, X_train, y_train):
    n_estimators = trial.suggest_int("n_estimators", 100, 800)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=1
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    return -score.mean()

def predict_current_price_using_rf(option_type, ticher):
  for proxy in feature_combinations:
    train_dataset = dataset[ticker]["train"]
    train_dataset = train_dataset[train_dataset["type"] == option_type]
    X_train = train_dataset[list_histos_datas_inputs + proxy].values
    y_train = train_dataset[["lastPrice"]].values

    test_dataset = dataset[ticker]["test"]
    test_dataset = test_dataset[test_dataset["type"] == option_type]
    X_test = test_dataset[list_histos_datas_inputs+ proxy].values
    y_test = test_dataset[["lastPrice"]].values

    #Normaliser les dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    study = study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: objective_for_rf(trial, X_train, y_train), n_trials=10)

    best_params = study.best_params
    #print(best_params)

    mae, rmse, nrmse = train_rf_with_optuna(
        X_train, y_train,
        X_test, y_test,
        best_params
    )
    print(f"{proxy} for {ticker} => (MAE={mae:.3f}; RMSE={rmse:.3f}; ; NRMSE={nrmse:.3f})")

