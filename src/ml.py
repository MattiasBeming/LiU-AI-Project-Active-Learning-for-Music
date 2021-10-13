import numpy as np  # Version 1.19.1
import pandas as pd
import warnings


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
    if (path_arousal.is_file() and path_valence.is_file()):
        df1 = pd.read_csv(path_arousal, delimiter=",")
        df2 = pd.read_csv(path_valence,  delimiter=",")
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
        return(covs)
    else:
        warnings.warn(
            "Warning: One of the file paths is incorrect.")
    return(-1)
