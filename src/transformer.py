import numpy as np  # Version 1.19.1
from pathlib import Path


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
    data = np.empty((92, 6670, 1))
    found_inds = []
    for i in file_indexes:
        file_path = Path(
            f"./../../emo-music-features/default_features/{i}.csv")
        if file_path.is_file():
            found_inds.append(i)
            data_temp = np.genfromtxt(file_path, delimiter=';')
            data_temp = data_temp.reshape(
                (data_temp.shape[0], data_temp.shape[1], 1))
            data = np.concatenate((data, data_temp), axis=2)

    return(data, found_inds)


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


# Run code
data, inds = load_songs(range(1, 3))
print(np.shape(data))
print(len(find_inds()))
