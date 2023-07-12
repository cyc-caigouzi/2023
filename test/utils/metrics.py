import numpy as np

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true)) * 100

def R2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)

def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    mape = MAPE(pred, true)
    r_squared = R2(pred, true)

    return mse, mae, mape, r_squared
