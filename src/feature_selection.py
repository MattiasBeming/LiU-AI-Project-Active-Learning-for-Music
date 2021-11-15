import os
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np


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


def scale(tr, te):
    """
    Scale the data before performing PCA.
    """
    scaler = StandardScaler()
    tr.iloc[:, 2:].values[:] = scaler.fit_transform(tr.iloc[:, 2:].values[:])
    te.iloc[:, 2:].values[:] = scaler.transform(te.iloc[:, 2:].values[:])
    return tr, te


def pca(tr, te, percent):
    """
    Perform PCA.

    Args:
        percent (float): perform pca to describe 'percent' of the data.
    """
    pca = PCA(n_components=percent, svd_solver='full')
    tr_pca = pca.fit_transform(tr.iloc[:, 2:].values[:])
    te_pca = pca.transform(te.iloc[:, 2:].values[:])

    print((f"\n{pca.n_components_} number of features "
           f"holds {percent} of the Data."))

    return tr_pca, te_pca


def feature_selection(filepath, tr_size=0.8, pca_percent=0.99):
    """
    Split the data into training and test, and perform PCA on the data.

    Args:
        filepath (Path): -
        tr_size (float, optional): Determines the size of train split.
        pca_percent (float, optional): Determines n_components for pca,
                                       how much variance does the
                                       components hold.
    """
    data = pd.read_csv(filepath, index_col=[0], header=[0, 1, 2])

    print(f"Shape of data before split: {data.shape}")
    tr, te = split_data(data, tr_size)

    tr, te = scale(tr, te)

    # Saves song_id and sample_id columns
    tr_idx = np.array(tr.iloc[:, 0:2].values[:])
    te_idx = np.array(te.iloc[:, 0:2].values[:])

    print(f"Shape pre PCA: tr shape: {tr.shape}, te shape: {te.shape}")
    tr, te = pca(tr, te, pca_percent)

    # Add song_id and sample_id columns to training and test data
    tr = np.hstack((tr_idx, tr))
    te = np.hstack((te_idx, te))

    # Cast first 2 columns with song_id and sample_id, to int
    tr[:, 0:2] = tr[:, 0:2].astype('int')
    te[:, 0:2] = te[:, 0:2].astype('int')

    print((f"Shape post PCA with {pca_percent} variance, "
          f"tr shape: {tr.shape}, te shape: {te.shape}"))

    file = str(filepath)[:-4]
    np.save(Path(file + "_train_pca.npy"), tr)
    np.save(Path(file + "_test_pca.npy"), te)
    return


def main():
    os.chdir('./../')  # Change to parent directory
    filepath = Path("data/features_librosa.csv")
    feature_selection(filepath, 0.8, 0.99)


if __name__ == "__main__":
    main()
