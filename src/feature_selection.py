import os
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# OBS, Run this file from src


class Method(Enum):
    D = 0  # Default methods - no feature selection
    PCA = 1
    VT = 2


def split_data(data, tr_size):
    """
    Splits data into a training and test set based on tr_size
    """
    unique_songs = np.unique(np.array(data["song_id"]))

    nr_tr = int(np.ceil(len(unique_songs)*tr_size))
    tr_songs = np.random.choice(unique_songs, nr_tr, replace=False)
    te_songs = np.setdiff1d(unique_songs, tr_songs, assume_unique=True)

    ids_tr = []
    for id_ in tr_songs:
        ids = list(data[np.array(data["song_id"] == id_)].index)
        ids_tr += ids

    ids_te = []
    for id_ in te_songs:
        ids = list(data[np.array(data["song_id"] == id_)].index)
        ids_te += ids

    tr = data.iloc[ids_tr]
    tr.index = pd.RangeIndex(len(tr.index))

    te = data.iloc[ids_te]
    te.index = pd.RangeIndex(len(te.index))

    return tr, te


def scale(tr, te, before=True):
    """
    Scale the data before/after performing feature selection.
    """
    scaler = StandardScaler()
    if before:
        tr.iloc[:, 2:].values[:] = scaler.fit_transform(
            tr.iloc[:, 2:].values[:])
        te.iloc[:, 2:].values[:] = scaler.transform(te.iloc[:, 2:].values[:])
        return tr, te
    tr = scaler.fit_transform(tr)
    te = scaler.transform(te)
    return tr, te


def pca(tr, te, percent):
    """
    Perform PCA.

    Args:
        percent (float): perform pca to describe 'percent' of the data.
    """
    pca = PCA(n_components=percent, svd_solver='full')
    tr = pca.fit_transform(tr.iloc[:, 2:].values[:])
    te = pca.transform(te.iloc[:, 2:].values[:])

    print((f"\n{pca.n_components_} number of features "
           f"holds {percent} of the Data."))

    return tr, te


def variance_threshold(tr, te, threshold):
    """
    Perform variance threshold. Removes features based on columns-wise
    variance.

    Args:
        threshold (float): if variance <= threshold removes that feature
    """
    selector = VarianceThreshold(threshold=threshold)
    tr_vt = selector.fit_transform(tr.iloc[:, 2:].values[:])
    te_vt = selector.transform(te.iloc[:, 2:].values[:])

    print((f"\n{tr_vt.shape[1]} number of features "
           f"has higher variance than threshold: {threshold}"))

    return tr_vt, te_vt


def feature_selection(filepath, tr_size=0.8, method=Method.PCA,
                      pca_percent=0.99, threshold=100):
    """
    Split the data into training and test, and perform PCA on the data.

    Args:
        filepath (Path): -
        tr_size (float, optional): Determines the size of train split.
        pca_percent (float, optional): Determines n_components for pca,
                                       how much variance does the
                                       components hold.
    """
    print(f"Running method: {method.name}...")
    data = pd.read_csv(filepath, index_col=[0], header=[0, 1, 2])

    print(f"Shape of data before split: {data.shape}")
    tr, te = split_data(data, tr_size)

    tr, te = scale(tr, te)

    # Saves song_id and sample_id columns
    tr_idx = np.array(tr.iloc[:, 0:2].values[:])
    te_idx = np.array(te.iloc[:, 0:2].values[:])

    if method == Method.PCA:
        print(f"Shape pre PCA: tr shape: {tr.shape}, te shape: {te.shape}")
        tr, te = pca(tr, te, pca_percent)
        tr, te = scale(tr, te, before=False)

    if method == Method.VT:
        print(f"Shape pre VT: tr shape: {tr.shape}, te shape: {te.shape}")
        tr, te = variance_threshold(tr, te, threshold)

    # Add song_id and sample_id columns to training and test data
    tr = np.hstack((tr_idx, tr))
    te = np.hstack((te_idx, te))

    # Cast first 2 columns with song_id and sample_id, to int
    tr[:, 0:2] = tr[:, 0:2].astype('int')
    te[:, 0:2] = te[:, 0:2].astype('int')

    print((f"Shape post selection: "
           f"tr shape: {tr.shape}, te shape: {te.shape}"))
    file = str(filepath)[:-4]
    np.save(Path(file + f"_train_{method.name}.npy"), tr)
    np.save(Path(file + f"_test_{method.name}.npy"), te)
    return


def main():
    os.chdir('./../')  # Change to parent directory
    filepath = Path("data/features_librosa.csv")
    feature_selection(filepath, 0.8, Method.PCA, 0.99, 100)


if __name__ == "__main__":
    main()
