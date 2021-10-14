import numpy as np  # Version 1.19.1
from pathlib import Path
import pandas as pd
import warnings


def load_songs(file_indexes=range(1, 1000)):
    """
    Loads all songs into an object.
    Input:
        file_indexes: A list of the indexes of the files to be loaded.
                Defaults to all indexes.
    Output:
        data: An 3D numpy array containing the song data.
        found_inds: A list of all the indexes from 'file_indexes' that has a
                corresponding data file.
                (Not all number from 1 to 1000 have a corresponding file.)
    """
    found_inds = []
    first = True
    for i in file_indexes:
        file_path = Path(
            f"./../../emo-music-features/default_features/{i}.csv")
        if file_path.is_file():
            found_inds.append(i)
            data_temp = np.genfromtxt(file_path, delimiter=';', skip_header=1)
            data_temp = data_temp.reshape(
                (1, data_temp.shape[0], data_temp.shape[1]))
            if first:
                data = data_temp
                first = False
            else:
                data = np.concatenate((data, data_temp), axis=0)

    return(data, found_inds)


def load_annotations(path_arousal, path_valence):
    """
    Loads in valence and arousal labels from respective paths.
    Input:
        path_arousal: A Path object (from pathlib), containing
            the path to "arousal_cont_average.csv".
        path_valence: A Path object (from pathlib), containing
            the path to "valence_cont_average.csv".
    Output:
        arousal: A 2D pandas object.
        valence: A 2D pandas object.
    """
    if (path_arousal.is_file() and path_valence.is_file()):
        arousal = pd.read_csv(path_arousal, delimiter=",")
        valence = pd.read_csv(path_valence,  delimiter=",")
        return (arousal, valence)
    else:
        warnings.warn(
            "Warning: One or both of the file paths are incorrect.")
        return(-1)


def find_inds(file_indexes=range(1, 1000)):
    """
    Loads all songs into an object.
    Input:
        file_indexes: A list of the indexes of the files to be loaded.
                    Defaults to all indexes.
    Output:
        found_inds: A list of all the indexes from 'file_indexes' that has a
                    corresponding data file.
                    (Not all number from 1 to 1000 have a corresponding file.)
    """
    found_inds = []
    for i in file_indexes:
        file_path = Path(
            f"./../../emo-music-features/default_features/{i}.csv")
        if file_path.is_file():
            found_inds.append(i)
    return found_inds
