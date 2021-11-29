import numpy as np  # Version 1.19.1
import pandas as pd
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from transformer import *
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


def k_neighbors(X, y, k, train=True):
    """
    Creates a KNeighborsRegressor

    Args:
        X (np.array): Feature data from labeled data set, as a numpy array with
                format (num_songs*time_samples, num_features)
        y (np.array): Arousal/valence annotations belonging to the songs and
                timestamps in X. 2D numpy array with
                num_songs*time_samples rows.
        train (bool, optional): True if model should fit to data.
                Defaults to True.
    Returns:
        KNeighborsRegressor: Decision tree regressor model.
    """
    if train:
        return KNeighborsRegressor(n_neighbors=k).fit(X, y)
    return KNeighborsRegressor(n_neighbors=k)


def linear_regression(X, y, train=True):
    """
    Creates a LinearRegression regressor

    Args:
        X (np.array): Feature data from labeled data set, as a numpy array with
                format (num_songs*time_samples, num_features)
        y (np.array): Arousal/valence annotations belonging to the songs and
                timestamps in X. 2D numpy array with
                num_songs*time_samples rows.
        train (bool, optional): True if model should fit to data.
                Defaults to True.
    Returns:
        LinearRegression: Decision tree regressor model.
    """
    if train:
        return LinearRegression().fit(X, y)
    return LinearRegression()


def decision_tree(X, y, train=True):
    """
    Creates a decision tree regressor

    Args:
        X (np.array): Feature data from labeled data set, as a numpy array with
                format (num_songs*time_samples, num_features)
        y (np.array): Arousal/valence annotations belonging to the songs and
                timestamps in X. 2D numpy array with
                num_songs*time_samples rows.
        train (bool, optional): True if model should fit to data.
                Defaults to True.
    Returns:
        DecisionTreeRegressor: Decision tree regressor model.
    """
    if train:
        return DecisionTreeRegressor().fit(X, y)
    return DecisionTreeRegressor()


def gradient_tree_boosting(X, y, train=True):
    """
    Creates a gradient tree boosting regressor

    Args:
        X (np.array): Feature data from labeled data set, as a numpy array with
                format (num_songs*time_samples, num_features)
        y (np.array): Arousal/valence annotations belonging to the songs and
                timestamps in X. 2D numpy array with
                num_songs*time_samples rows.
        train (bool, optional): True if model should fit to data.
                Defaults to True.
    Returns:
        DecisionTreeRegressor: gradient tree boosting model.
    """
    if train:
        return MultiOutputRegressor(GradientBoostingRegressor()).fit(X, y)
    return MultiOutputRegressor(GradientBoostingRegressor())


def ensemble(X, y, regressors, names, train=True):
    """
    Creates a voting ensemble  regressor

    Args:
        X (np.array): Feature data from labeled data set, as a numpy array with
                format (num_songs*time_samples, num_features)
        y (np.array): Arousal/valence annotations belonging to the songs and
                timestamps in X. 2D numpy array with
                num_songs*time_samples rows.
        regressors (list): List of regressors from sklearn.
        names (list): List of names for the regresssors.
                Each regressor must have a name.
        train (bool, optional): True if model should fit to data.
                Defaults to True.
    Returns:
        VotingRegressor: VotingRegressor model.
    """
    assert len(regressors) == len(names)
    est = list(zip(names, regressors))
    if train:
        return MultiOutputRegressor(VotingRegressor(estimators=est)).fit(X, y)
    return MultiOutputRegressor(VotingRegressor(estimators=est))


def correlation(path_arousal, path_valence):
    """
    Calculates the covariances between the dynamic annotations
        for arousal and valence for all songs.
    Input:
        path_arousal: A Path object (from pathlib), containing
            the path to "arousal_cont_average.csv".
        path_valence: A Path object (from pathlib), containing
            the path to "valence_cont_average.csv".
    Output:
        covs: A 1D numpy array containing the dynamic covariances
            for each song. If the file paths were incorrect, this
            will be set to -1.
    """
    if path_arousal.is_file() and path_valence.is_file():
        df1 = pd.read_csv(path_arousal, delimiter=",")
        df2 = pd.read_csv(path_valence, delimiter=",")
        covs = np.zeros(df1.shape[0]-1)
        max = df1.shape[0]
        for i in range(1, max):
            X = pd.concat([df1.iloc[i, 1:], df2.iloc[i, 1:]], axis='columns')
            X.columns = ["arousal", "valence"]
            cov = np.corrcoef(np.transpose(X))
            covs[i-1] = cov[1, 0]
        print(np.mean(np.abs(covs[0:max-1])))
        # 0.5212059045950358
        print(np.mean(covs[0:max-1]))
        # 0.2215276019990472
        return covs
    else:
        warnings.warn(
            "Warning: One of the file paths is incorrect.")
    return -1
