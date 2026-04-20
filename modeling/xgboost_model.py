import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor

def train_xgboost_with_optuna(X_train, y_train, X_test, y_test, best_params):

    params = best_params.copy()
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"
    params["tree_method"] = "hist"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    nrmse = rmse / np.mean(y_test)

    return mae, rmse, nrmse


def objective_for_xgboost(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5)
    }

    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")

    return -score.mean()

def predict_current_price_using_xgboost(option_type, ticher):
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
    study.optimize(lambda trial: objective_for_xgboost(trial, X_train, y_train), n_trials=10)
    best_params = study.best_params
    #print(best_params)

    mae, rmse, nrmse = train_xgboost_with_optuna(
        X_train, y_train,
        X_test, y_test,
        best_params
    )
    print(f"{proxy} for {ticker} => (MAE={mae:.3f}; RMSE={rmse:.3f}; ; NRMSE={nrmse:.3f})")

