import numpy as np
import pandas as pd
from scipy import spatial


def max_sampling_helper_(values, song_ids, batchsize, n_samples):
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
        (list): song_ids which needs to be labelled
    """
    indexes = np.unique(song_ids, return_index=True)[1]
    song_ids_unique = [int(song_ids[index]) for index in sorted(indexes)]
    error_msg = f'Batchsize({batchsize} > nr songs({len(song_ids_unique)})'
    assert batchsize <= len(song_ids_unique), error_msg
    values_song = [np.sum(values[start:start+n_samples])
                   for start in range(0, len(values), n_samples)]
    print(values_song)
    print(song_ids_unique)
    idx = np.argpartition(values_song, -batchsize)[-batchsize:]
    return [song_ids_unique[i] for i in idx]


def max_std_sampling(pred, song_ids, batchsize, n_samples):
    """
    max_std_sampling implements the query by committee informativeness
    measurement.
    Indexes batchsize song_ids with the predictions that have the
    highest standard deviation

    Args:
        pred (np.ndarray): NxKxT 3d-Matrix containing predictions from every
            regressor for every sample, where N is number of samples in
            total. k is the amount of regressors, T is targets
        song_ids (list/np.array): song_ids, must be ordered correctly to pred
        batchsize (int): Amount of samples to be indexed for labelling
            each epoch.
            Note: batchsize =< unique song ids !
        n_samples (int): number of samples in each song

    Returns:
        (list): song_ids which needs to be labelled
    """
    stds = np.std(pred, axis=(1, 2))
    return max_sampling_helper_(stds, song_ids, batchsize, n_samples)


def output_greedy_sampling(labeled_data, unlabeled_output,
                           song_ids, batchsize, n_samples):
    """
    Performs greedy sampling on outputs

    Args:
        labeled_data (np.ndarray): targets for arousal and valence
        unlabeled_output (np.ndarray): predictions for arousal and valence
            from ML model
        song_ids (list/np.array): song_ids, must be ordered correctly to pred
        batchsize (int): Amount of samples to be indexed for labelling
            each epoch.
            Note: batchsize =< unique song ids !
        n_samples (int): number of samples in each song

    Returns:
        (list): song_ids which needs to be labelled
    """
    # Compute the distance from all unlabeled output data to the closest
    # labeled data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_output, labeled_data), axis=1)
    # Then select the batch_size number of songs that have the largest
    # distance to label and return song_ids
    return max_sampling_helper_(min_distances, song_ids, batchsize, n_samples)


def input_greedy_sampling(labeled_feat, unlabeled_feat,
                          song_ids, batchsize, n_samples):
    """
    Performs greedy sampling on features(inputs)

    Args:
        labeled_feat (panda.Dataframe): features used for training
        unlabeled_feat (panda.Dataframe): candidate features to be labelled
        song_ids (list/np.array): list of song ids
            ordered correctly to unlabeled_feat
        batchsize (int): Amount of samples to be indexed for labelling
            each epoch. Note: batchsize =< unique song ids !
        n_samples (int): number of samples in each song

    Returns:
        (list): song_ids which needs to be labelled
    """
    # Compute the distance from all unlabeled feature data to
    # closest labeled feature data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_feat, labeled_feat), axis=1)

    # Then select the batch_size number of samples that have the largest
    # distance to label features and return indices
    return max_sampling_helper_(min_distances, song_ids, batchsize, n_samples)


def input_output_greedy_sampling(labeled_feat, unlabeled_feat,
                                 labeled_data, unlabeled_output,
                                 song_ids, batchsize, n_samples):
    """
    Performs greedy sampling on features(inputs) and outputs

    Args:
        labeled_feat (panda.Dataframe): features used for training
        unlabeled_feat (panda.Dataframe): candidate features to be labelled
        labeled_data (np.ndarray): targets for arousal and valence
        unlabeled_output (np.ndarray): predictions for arousal and valence
            from ML model
        song_ids (list/np.array): list of song ids
            ordered correctly to unlabeled_feat
        batchsize (int): Amount of samples to be indexed for labelling
            each epoch. Note: batchsize =< unique song ids !
        n_samples (int): number of samples in each song

    Returns:
        (list): song_ids which needs to be labelled
    """

    # Compute the distance from all unlabeled feature data to
    # closest labeled feature data point and assign minimum one to each point
    min_distances_in = np.min(spatial.distance.cdist(
        unlabeled_feat, labeled_feat), axis=1)

    # Compute the distance from all unlabeled output data to the closest
    # labeled data point and assign minimum one to each point
    min_distances_out = np.min(spatial.distance.cdist(
        unlabeled_output, labeled_data), axis=1)
    # Choose min at each index comparing both arrays
    min_distances = np.minimum(min_distances_in, min_distances_out)

    # Then select the batch_size number of samples that have the largest
    # distance and return indices
    return max_sampling_helper_(min_distances, song_ids, batchsize, n_samples)
