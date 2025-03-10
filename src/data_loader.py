import numpy as np
import pandas as pd

def load_dataset(k):
    """
    Load training and test data for a given k (0,1,2).

    Args:
        k (int): Dataset index (0,1,2).

    Returns:
        X_train (ndarray): Training features.
        Y_train (ndarray): Training labels.
        X_test (ndarray): Test features.
    """
    X_train = pd.read_csv(f"./data/Xtr{k}.csv", index_col=0)
    Y_train = pd.read_csv(f"./data/Ytr{k}.csv", index_col=0)
    X_test = pd.read_csv(f"./data/Xte{k}.csv", index_col=0)

    X_train = np.array(X_train).squeeze()
    Y_train = np.array(Y_train).squeeze()
    X_test = np.array(X_test).squeeze()

    return X_train, Y_train, X_test

