import numpy as np


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
