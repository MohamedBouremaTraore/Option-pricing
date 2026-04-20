import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_ffnn(best_params, n_features):
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU
    }

    activation = activations[best_params["activation"]]
    n_layers = best_params["n_layers"]

    layers = []
    in_dim = n_features

    for i in range(n_layers):
        out_dim = best_params[f"n_units_{i}"]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())


        in_dim = out_dim

    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)

def train_ffnn_with_optuna(X_train, y_train, X_test, y_test, best_params, epochs=50):

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

    model = build_ffnn(best_params, X_train.shape[1])

    optimizer = getattr(optim, best_params["optimizer"])(
        model.parameters(),
        lr=best_params["lr"]
    )

    loss_fn = nn.MSELoss()

    # --------- Training ----------
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    # --------- Prediction ----------
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    nrmse = rmse / np.mean(y_test)

    return mae, rmse, nrmse


def create_model_for_ffnn(trial, n_features):

    n_layers = trial.suggest_int("n_layers", 2, 10)# nombre de hidden layers

    activation_name = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid", "elu", "selu"] # choix de la fonction d’activation
    )

    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU
    }
    activation = activations[activation_name]

    layers = []
    in_dim = n_features

    for i in range(n_layers):
        out_dim = trial.suggest_int(f"n_units_{i}", 2, 128)

        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())

        in_dim = out_dim

    # output layer
    layers.append(nn.Linear(in_dim, 1))

    return nn.Sequential(*layers)


# ---- objective ----
def objective_for_ffnn(trial, datasetForOptimization, X_train, y_train):

    model = create_model_for_ffnn(trial, X_train.shape[1])

    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "SGD"] #optimizer
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) #learning_rate

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) #batch_size

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    loader = DataLoader(datasetForOptimization, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()

    # ---- training ----
    for epoch in range(40):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    return loss.item()

def predict_current_price_using_ffnn(option_type, ticher):
  for proxy in feature_combinations:
    #Prepare training dataset
    train_dataset = dataset[ticker]["train"]
    train_dataset = train_dataset[train_dataset["type"] == option_type]
    X_train = train_dataset[list_histos_datas_inputs + proxy].values
    y_train = train_dataset[["lastPrice"]].values

    #Prepare test dataset
    test_dataset = dataset[ticker]["test"]
    test_dataset = test_dataset[test_dataset["type"] == option_type]
    X_test = test_dataset[list_histos_datas_inputs+ proxy].values
    y_test = test_dataset[["lastPrice"]].values

    #Normaliser les dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Determine best architecture and hyper-parameters
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    datasetForOptimization = TensorDataset(X_tensor, y_tensor)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: objective_for_ffnn(trial, datasetForOptimization, X_train, y_train), n_trials=10)
    best_params = study.best_params
    best_params = study.best_params
    #print(best_params)

    mae, rmse, nrmse = train_ffnn_with_optuna(
        X_train, y_train,
        X_test, y_test,
        best_params
    )
    print(f"{proxy} for {ticker} => (MAE={mae:.3f}; RMSE={rmse:.3f}; ; NRMSE={nrmse:.3f})")

