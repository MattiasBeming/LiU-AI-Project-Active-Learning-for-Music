from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Constants (paths are relative to python working directory).
# See README.md for more info.
FEATURES_LIBROSA_PATH = "data/features_librosa_yyyy-MM-dd_hh.mm.ss.csv"
TRVA_SIZE = 0.8
VA_SIZE = 0.25
PCA_PERCENT = 0.99
VT_THRESHOLD = 100


class Method(Enum):
    D = 0  # Default methods - no feature selection
    PCA = 1
    VT = 2


def split_data(data, trva_size):
    """
    Splits data intro trva and te, based on song_ids
    Args:
        data (pd.DataFrame): Dataframe containing all the feature data
        trva_size (float): percent size of data to be split into trva

    Returns:
        (pd.DataFrame, pd.DataFrame): trva and test dataframe
    """
    unique_songs = np.unique(np.array(data["song_id"]))

    nr_trva = int(np.ceil(len(unique_songs)*trva_size))
    np.random.seed(42069)
    trva_songs = np.random.choice(unique_songs, nr_trva, replace=False)
    te_songs = np.setdiff1d(unique_songs, trva_songs, assume_unique=True)

    ids_trva = []
    for id_ in trva_songs:
        ids = list(data[np.array(data["song_id"] == id_)].index)
        ids_trva += ids

    ids_te = []
    for id_ in te_songs:
        ids = list(data[np.array(data["song_id"] == id_)].index)
        ids_te += ids

    trva = data.iloc[ids_trva]
    trva.index = pd.RangeIndex(len(trva.index))

    te = data.iloc[ids_te]
    te.index = pd.RangeIndex(len(te.index))

    return trva, te


def split_trva(trva, va_size=0.25):
    """
    Splits data intro tr and va, based on song_ids
    Args:
        trva (np.ndarray): Dataframe containing all the feature data
        va_size (float): percent size of data to be split into va
    Returns:
        (np.ndarray, np.ndarray): tr and va dataframe
    """
    song_ids = trva[:, 0]
    unique_songs = np.unique(song_ids)

    nr_tr = int(np.ceil(len(unique_songs)*(1-va_size)))
    np.random.seed(42069)
    tr_songs = np.random.choice(unique_songs, nr_tr, replace=False)
    va_songs = np.setdiff1d(unique_songs, tr_songs, assume_unique=True)

    ids_tr = np.concatenate(
        [np.where(song_ids == id_) for id_ in tr_songs], axis=None)

    ids_va = np.concatenate(
        [np.where(song_ids == id_) for id_ in va_songs], axis=None)

    return trva[ids_tr], trva[ids_va]


def scale(trva, te, before=True):
    """
    Scale the data before/after performing feature selection.
    """
    scaler = StandardScaler()
    if before:
        trva.iloc[:, 2:].values[:] = scaler.fit_transform(
            trva.iloc[:, 2:].values[:])
        te.iloc[:, 2:].values[:] = scaler.transform(te.iloc[:, 2:].values[:])
        return trva, te
    trva = scaler.fit_transform(trva)
    te = scaler.transform(te)
    return trva, te


def pca(trva, te, percent):
    """
    Perform PCA.

    Args:
        percent (float): perform pca to describe 'percent' of the data.
    """
    pca = PCA(n_components=percent, svd_solver='full')
    trva = pca.fit_transform(trva.iloc[:, 2:].values[:])
    te = pca.transform(te.iloc[:, 2:].values[:])

    print((f"\n{pca.n_components_} number of features "
           f"holds {percent} of the Data."))

    return trva, te


def variance_threshold(trva, te, threshold):
    """
    Perform variance threshold. Removes features based on columns-wise
    variance.

    Args:
        threshold (float): if variance <= threshold removes that feature
    """
    selector = VarianceThreshold(threshold=threshold)
    trva_vt = selector.fit_transform(trva.iloc[:, 2:].values[:])
    te_vt = selector.transform(te.iloc[:, 2:].values[:])

    print((f"\n{trva_vt.shape[1]} number of features "
           f"has higher variance than threshold: {threshold}"))

    return trva_vt, te_vt


def feature_selection(filepath, trva_size=0.8, va_size=0.25, method=Method.PCA,
                      pca_percent=0.99, threshold=100):
    """
    Performs feature selection given method, applies split and saves
    resulting data to disk (.npy format)

    Args:
        filepath (pathlib.Path): Path to data (.csv file)
        trva_size (float, optional): trva percentage of data. Defaults to 0.8.
        va_size (float, optional): va percentage(of trva). Defaults to 0.25.
        method (Method(enum), optional): Determines selection method.
            Defaults to Method.PCA.
        pca_percent (float, optional): pca percentage of variance.
            Defaults to 0.99.
        threshold (int, optional): Variance threshold for vt method.
            Defaults to 100.
    """
    print(f"Running method: {method.name}...")
    data = pd.read_csv(filepath, index_col=[0], header=[0, 1, 2])

    print(f"Shape of data before split: {data.shape}")
    trva, te = split_data(data, trva_size)

    trva, te = scale(trva, te)

    # Saves song_id and sample_id columns
    trva_idx = np.array(trva.iloc[:, 0:2].values[:])
    te_idx = np.array(te.iloc[:, 0:2].values[:])

    if method == Method.PCA:
        print(f"Shape pre PCA: trva shape: {trva.shape}, te shape: {te.shape}")
        trva, te = pca(trva, te, pca_percent)
        trva, te = scale(trva, te, before=False)

    if method == Method.VT:
        print(f"Shape pre VT: trva shape: {trva.shape}, te shape: {te.shape}")
        trva, te = variance_threshold(trva, te, threshold)

    # Add song_id and sample_id columns to trvaaining and test data
    trva = np.hstack((trva_idx, trva))
    te = np.hstack((te_idx, te))

    # Cast first 2 columns with song_id and sample_id, to int
    trva[:, 0:2] = trva[:, 0:2].astype('int')
    te[:, 0:2] = te[:, 0:2].astype('int')

    file = str(filepath)[:-4]

    tr, va = split_trva(trva, va_size=va_size)

    print((f"Shape post selection: "
           f"tr shape: {tr.shape}, va shape: {va.shape}"
           f" trva shape: {trva.shape}, te shape: {te.shape}"))
    np.save(Path(file + f"_train_{method.name}.npy"), tr)
    np.save(Path(file + f"_val_{method.name}.npy"), va)
    np.save(Path(file + f"_trainval_{method.name}.npy"), trva)
    np.save(Path(file + f"_test_{method.name}.npy"), te)
    return


def main():
    filepath = Path(FEATURES_LIBROSA_PATH)
    feature_selection(filepath, TRVA_SIZE, VA_SIZE,
                      Method.PCA, PCA_PERCENT, None)
    feature_selection(filepath, TRVA_SIZE, VA_SIZE,
                      Method.VT, None, VT_THRESHOLD)


if __name__ == "__main__":
    main()
