import numpy as np  # Version 1.19.1
import pandas as pd
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from transformer import *
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from storage import Dataset


def neural_network(ds: Dataset, hyper_parameters: dict = {}, train=True):
    """
       Creates a neural network using sklearn's MLPRegressor.

       Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.
        train (bool, optional): If true, fit to ds. Defaults to True.

       Returns:
           [MLPRegressor]: A new MLPRegressor estimator.
       """
    if train:
        return MLPRegressor(**hyper_parameters).fit(
            ds.get_data(), ds.get_labels())
    return MLPRegressor(**hyper_parameters)


def k_neighbors(ds: Dataset, hyper_parameters: dict = {}, train=True):
    """
    Construct a KNeighborsRegressor model.

    Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.
        train (bool, optional): If true, fit to ds. Defaults to True.

    Returns:
        [KNeighborsRegressor]: A new KNeighborsRegressor estimator.
    """
    if train:
        return KNeighborsRegressor(**hyper_parameters).fit(
            ds.get_data(), ds.get_labels())
    return KNeighborsRegressor(**hyper_parameters)


def linear_regression(ds: Dataset, hyper_parameters: dict = {}, train=True):
    """
    Construct a LinearRegression model.

    Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.
        train (bool, optional): If true, fit to ds. Defaults to True.

    Returns:
        [LinearRegression]: A new LinearRegression estimator.
    """
    if train:
        return LinearRegression(**hyper_parameters).fit(ds.get_data(),
                                                        ds.get_labels())
    return LinearRegression(**hyper_parameters)


def decision_tree(ds: Dataset, hyper_parameters: dict = {}, train=True):
    """
    Construct a DecisionTreeRegressor model.

    Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.
        train (bool, optional): If true, fit to ds. Defaults to True.

    Returns:
        [DecisionTreeRegressor]: A new DecisionTreeRegressor estimator.
    """
    if train:
        return DecisionTreeRegressor(**hyper_parameters).fit(ds.get_data(),
                                                             ds.get_labels())
    return DecisionTreeRegressor(**hyper_parameters)


def gradient_tree_boosting(ds: Dataset, hyper_parameters: dict = {},
                           train=True):
    """
    Construct a MultiOutputRegressor(GradientBoostingRegressor) model.

    Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.
        train (bool, optional): If true, fit to ds. Defaults to True.

    Returns:
        [MultiOutputRegressor(GradientBoostingRegressor)]: A new
            MultiOutputRegressor(GradientBoostingRegressor) estimator.
    """
    if train:
        return MultiOutputRegressor(
            GradientBoostingRegressor(**hyper_parameters)
        ).fit(ds.get_data(), ds.get_labels())
    return MultiOutputRegressor(GradientBoostingRegressor(**hyper_parameters))


def ensemble(ds: Dataset, hyper_parameters: dict = {}, train=True):
    """
    Construct a MultiOutputRegressor(VotingRegressor) model.

    Args:
        ds (Dataset): Dataset to fit on. May be `None` if `train=False`.
        hyper_parameters (dict): Should contain all desired hyperparameters.
            These will be forwarded as keyword arguments to the returned
            regressor instance. Defaults to empty dict.

            There are two required hyper parameters: a list of regressors to
            use named "regressors", and a list of matching regressor names
            called "names".
        train (bool, optional): If true, fit to ds. Defaults to True.

    Returns:
        [MultiOutputRegressor(VotingRegressor)]: A new
            MultiOutputRegressor(VotingRegressor) estimator.
    """
    regressors = hyper_parameters["regressors"]
    names = hyper_parameters["names"]

    assert len(regressors) == len(names)

    # Get all remaining hyper parameters
    remaining_hyper_parameters = {
        hp: hyper_parameters[hp]
        for hp in hyper_parameters
        if hp not in ("regressors", "names")
    }

    est = list(zip(names, regressors))
    if train:
        return MultiOutputRegressor(VotingRegressor(
            estimators=est,
            **remaining_hyper_parameters
        )).fit(ds.get_data(), ds.get_labels())
    return MultiOutputRegressor(VotingRegressor(
        estimators=est,
        **remaining_hyper_parameters
    ))


def correlation(path_arousal, path_valence):
    """
    Calculates the covariances between the dynamic annotations
        for arousal and valence for all songs.
    Args:
        path_arousal (pathlib.Path): A path to "arousal_cont_average.csv".
        path_valence (pathlib.Path): A path to "valence_cont_average.csv".

    Returns:
        [np.ndarray]: A 1D numpy array containing the dynamic covariances
            for each song. If the file paths were incorrect, this
            will be set to - 1.
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
