import numpy as np
import sklearn as sk
from scipy import spatial
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor
from api.storage import Dataset


def _max_sampling_helper(values, song_ids, batchsize, n_samples):
    """
    Compare sum of values for each song, finds the batchsize-max and return
    song_ids related to those set of values.

    Args:
        values (np.array): values that are to be compared
        song_ids (list/np.array): song_ids, must be ordered correctly to pred
        batchsize (int): Amount of samples to be indexed for labelling
            each epoch.
            Note: batchsize =< unique song ids !
        n_samples (int): number of samples in each song

    Returns:
        (list): song_ids which needs to be labelled.
    """
    indexes = np.unique(song_ids, return_index=True)[1]
    song_ids_unique = [int(song_ids[index]) for index in sorted(indexes)]
    error_msg = f'Batchsize({batchsize} > nr songs({len(song_ids_unique)})'
    assert batchsize <= len(song_ids_unique), error_msg
    values_song = [np.sum(values[start:start+n_samples])
                   for start in range(0, len(values), n_samples)]
    idx = np.argpartition(values_song, -batchsize)[-batchsize:]
    return [song_ids_unique[i] for i in idx]


def _get_ensemble_predictions(features, ensemble):
    target_estimators = ensemble.estimators_
    n_samples = features.shape[0]
    n_targets = len(target_estimators)
    n_estimators = len(target_estimators[0].estimators_)
    preds = np.zeros((n_samples, n_estimators, n_targets))
    for t, target_estimator in enumerate(target_estimators):
        preds[:, :, t] = target_estimator.transform(features)
    return preds


def max_std_sampling(labeled_ds: Dataset, unlabeled_ds: Dataset,
                     unlabeled_ds_predictions: np.ndarray,
                     batch_size: int, n_samples_per_song: int,
                     used_estimator: sk.base.BaseEstimator):
    """
    Performs query by committee. This function will perform multiple
    predictions, and may therefore take some time to complete.

    Args:
        labeled_ds (Dataset): Not used. Set to `None`.
        unlabeled_ds (Dataset): Unlabeled candidate data. The songs in this
            dataset are the ones from which some will be chosen for usage as
            training data.
        unlabeled_ds_predictions (np.ndarray): Not used. Set to `None`.
        batch_size (int): Number of samples to be indexed for labeling
            each epoch. Note: batch size must be smaller than or equal to the
            number of unlabeled song IDs.
        n_samples_per_song (int): Number of samples in each song.
        used_estimator (sklearn.base.BaseEstimator): A fitted
            `MultiOutputRegressor(VotingRegressor)` estimator whose composite
            estimators will be used for the committee.

    Raises:
        ValueError: When an invalid `used_estimator` was provided.

    Returns:
        (list): Song IDs from `unlabeled_ds` which need to be labeled.
    """

    # Check if used_estimator valid
    if not (isinstance(used_estimator, MultiOutputRegressor) and
            isinstance(used_estimator.estimator, VotingRegressor)):
        raise ValueError(
            "Invalid regressor. Must be MultiOutputRegressor(VotingRegressor)."
        )

    pred = _get_ensemble_predictions(
        unlabeled_ds.get_data(), used_estimator)

    stds = np.std(pred, axis=(1, 2))
    return _max_sampling_helper(
        stds,
        unlabeled_ds.get_contained_song_ids(),
        batch_size, n_samples_per_song
    )


def output_greedy_sampling(labeled_ds: Dataset, unlabeled_ds: Dataset,
                           unlabeled_ds_predictions: np.ndarray,
                           batch_size: int, n_samples_per_song: int,
                           used_estimator: sk.base.BaseEstimator):
    """
    Performs greedy sampling on labels/predictions (outputs).

    Args:
        labeled_ds (Dataset): Current labeled training data.
        unlabeled_ds (Dataset): Unlabeled candidate data. The songs in this
            dataset are the ones from which some will be chosen for usage as
            training data.
        unlabeled_ds_predictions (np.ndarray): Label predictions for the
            unlabeled dataset.
        batch_size (int): Number of samples to be indexed for labeling
            each epoch. Note: batch size must be smaller than or equal to the
            number of unlabeled song IDs.
        n_samples_per_song (int): Number of samples in each song.
        used_estimator (sklearn.base.BaseEstimator): Not used. Set to `None`.

    Returns:
        (list): Song IDs from `unlabeled_ds` which need to be labeled.
    """
    # Compute the distance from all unlabeled output data to the closest
    # labeled data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_ds_predictions, labeled_ds.get_labels()), axis=1)
    # Then select the batch_size number of songs that have the largest
    # distance to label and return song_ids
    return _max_sampling_helper(
        min_distances,
        unlabeled_ds.get_contained_song_ids(),
        batch_size, n_samples_per_song
    )


def input_greedy_sampling(labeled_ds: Dataset, unlabeled_ds: Dataset,
                          unlabeled_ds_predictions: np.ndarray,
                          batch_size: int, n_samples_per_song: int,
                          used_estimator: sk.base.BaseEstimator):
    """
    Performs greedy sampling on features (inputs).

    Args:
        labeled_ds (Dataset): Current labeled training data.
        unlabeled_ds (Dataset): Unlabeled candidate data. The songs in this
            dataset are the ones from which some will be chosen for usage as
            training data.
        unlabeled_ds_predictions (np.ndarray): Not used. Set to `None`.
        batch_size (int): Number of samples to be indexed for labeling
            each epoch. Note: batch size must be smaller than or equal to the
            number of unlabeled song IDs.
        n_samples_per_song (int): Number of samples in each song.
        used_estimator (sklearn.base.BaseEstimator): Not used. Set to `None`.

    Returns:
        (list): Song IDs from `unlabeled_ds` which need to be labeled.
    """
    # Compute the distance from all unlabeled feature data to
    # closest labeled feature data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_ds.get_data(), labeled_ds.get_data()), axis=1)

    # Then select the batch_size number of samples that have the largest
    # distance to label features and return indices
    return _max_sampling_helper(
        min_distances,
        unlabeled_ds.get_contained_song_ids(),
        batch_size, n_samples_per_song
    )


def input_output_greedy_sampling(labeled_ds: Dataset, unlabeled_ds: Dataset,
                                 unlabeled_ds_predictions: np.ndarray,
                                 batch_size: int, n_samples_per_song: int,
                                 used_estimator: sk.base.BaseEstimator):
    """
    Performs greedy sampling on features (inputs) and
    labels/predictions (outputs).

    Args:
        labeled_ds (Dataset): Current labeled training data.
        unlabeled_ds (Dataset): Unlabeled candidate data. The songs in this
            dataset are the ones from which some will be chosen for usage as
            training data.
        unlabeled_ds_predictions (np.ndarray): Label predictions for the
            unlabeled dataset.
        batch_size (int): Number of samples to be indexed for labeling
            each epoch. Note: batch size must be smaller than or equal to the
            number of unlabeled song IDs.
        n_samples_per_song (int): Number of samples in each song.
        used_estimator (sklearn.base.BaseEstimator): Not used. Set to `None`.

    Returns:
        (list): Song IDs from `unlabeled_ds` which need to be labeled.
    """

    # Compute the distance from all unlabeled feature data to
    # closest labeled feature data point and assign minimum one to each point
    min_distances_in = np.min(spatial.distance.cdist(
        unlabeled_ds.get_data(), labeled_ds.get_data()), axis=1)

    # Compute the distance from all unlabeled output data to the closest
    # labeled data point and assign minimum one to each point
    min_distances_out = np.min(spatial.distance.cdist(
        unlabeled_ds_predictions, labeled_ds.get_labels()), axis=1)
    # Choose min at each index comparing both arrays
    min_distances = np.minimum(min_distances_in, min_distances_out)

    # Then select the batch_size number of samples that have the largest
    # distance and return indices
    return _max_sampling_helper(
        min_distances,
        unlabeled_ds.get_contained_song_ids(),
        batch_size, n_samples_per_song
    )
