import numpy as np
import pandas as pd
from scipy import spatial


def max_std_sampling(pred, batchsize):
    """
    max_std_sampling implements the query by committee informativeness
    measurement.
    Indexes batchsize samples with the predictions that have the
    highest standard deviation(sd)

    Parameters:
    pred - NxKxT 3d-Matrix containing predictions from every regressor for
        every sample,
        where N is number of samples in total.
        k is the amount of regressors, T is targets
    batchsize - Amount of samples to be indexed for labelling each epoch,
        batchsize =< N

    Output:
    indexes - Vector of Indexes for samples in the data which are to be
    labelled (OBS: Not sorted)
    """
    error_msg = 'Batchsize must be smaller or equal to number of samples N'
    assert batchsize <= pred.shape[0], error_msg
    return np.argpartition(np.std(pred, axis=(1, 2)), -batchsize)[-batchsize:]


###########################################################################
# ----------------------- Output Greedy Sampling ------------------------ #
###########################################################################

unlabeled_output = np.array(
    [[243,  3173],
     [525,  2997]])

labeled_data = np.array(
    [[682, 2644],
     [277, 2651],
     [396, 2640]])


def output_greedy_sampling(labeled_data, unlabeled_output, batch_size):
    """
    :param labeled_data: Labeled data points from storage
    :param unlabeled_output: Unlabeled data points with predictions from ML
     model
    :param batch_size: Number of points to query label for
    :return: Indexes for the points to query on
    """

    # Compute the distance from all unlabeled output data to the closest
    # labeled data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_output, labeled_data), axis=1)

    # Then select the batch_size number of samples that have the largest
    # distance to label and return indices
    return np.argpartition(min_distances, -batch_size)[-batch_size:]


out_indices = output_greedy_sampling(labeled_data, unlabeled_output, 2)


###########################################################################
# ------------------------ Input Greedy Sampling ------------------------ #
###########################################################################

# Read feature data and get N columns to make data more manageable
feature_data = pd.read_csv(
    "~/Universitet/TDDE19/emo-music-features/default_features/2.csv", sep=";")
N = 3
labeled_feat = feature_data.iloc[0:3, 0:N]
unlabeled_feat = feature_data.iloc[4:7, 0:N]


def input_greedy_sampling(labeled_feat, unlabeled_feat, batch_size):
    """
    :param labeled_feat: Feature data from labeled set in ML
    :param unlabeled_feat: Feature data from unlabeled set in ML
    :param batch_size: Number of points to query label for
    :return: Indexes for the points to query on
    """
    # Compute the distance from all unlabeled feature data to
    # closest labeled feature data point and assign minimum one to each point
    min_distances = np.min(spatial.distance.cdist(
        unlabeled_feat, labeled_feat), axis=1)

    # Then select the batch_size number of samples that have the largest
    # distance to label features and return indices
    return np.argpartition(min_distances, -batch_size)[-batch_size:]


in_indices = input_greedy_sampling(labeled_feat, unlabeled_feat, 2)
