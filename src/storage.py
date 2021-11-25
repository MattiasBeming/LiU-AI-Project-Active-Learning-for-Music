from transformer import *
import warnings


# Global values
NO_OF_INITIAL_UNUSED_SAMPLES = 30
_datasets = {}


class Dataset():
    def __init__(self, data=None, labels_arousal=None, labels_valence=None,
                 contained_song_ids=[]):
        """
        Initialize non-empty dataset.

        Args:
            data (pandas.DataFrame):
                A 2Dpandas dataframe with dimensions
                    [[song_id, sample], features]
            labels_arousal (pandas.DataFrame):
                A 1D pandas dataframe.
            labels_valence (pandas.DataFrame):
                A 1D pandas dataframe.
            contained_song_ids (list):
                List containing indexes of loaded songs.
        """
        self._data = data
        self._labels_arousal = labels_arousal
        self._labels_valence = labels_valence
        self._contained_song_ids = contained_song_ids

    def get_data(self):
        """
        Returns the features of the dataset.

        Raises:
            ValueError: If dataset features is None.

        Returns:
            pandas.DataFrame: A 2D pandas dataframe. It has multi-indexed
            columns, first level being 'song_id' and second level 'sample_no'.
            Header indexes 'feature_no'.
        """
        if (self._data is not None):
            return self._data
        else:
            raise ValueError("Data was fetched before it was loaded. " +
                             "Do load_dataset() before calling get_dataset().")

    def get_arousal_labels(self):
        """
        Returns the labels for arousal for this dataset.

        Raises:
            ValueError: If 'labels_arousal' is None.

        Returns:
            pandas.DataFrame: A 1D pandas dataframe with labels for arousal.
        """
        if (self._labels_arousal is not None):
            return self._labels_arousal
        else:
            raise ValueError("Data was fetched before it was loaded. " +
                             "Do load_dataset() before calling get_dataset().")

    def get_valence_labels(self):
        """
        Returns the labels for valence for this dataset.

        Raises:
            ValueError: If 'labels_valence' is None.

        Returns:
            pandas.DataFrame: A 1D pandas dataframe with labels for valence.
        """
        if (self._labels_valence is not None):
            return self._labels_valence
        else:
            raise ValueError("Data was fetched before it was loaded. " +
                             "Do load_dataset() before calling get_dataset().")

    def get_labels(self):
        """
        Returns the concatinated labels for arousal and valence.

        Raises:
            ValueError: If 'labels_arousal' or 'labels_valence' is None.

        Returns:
            np.ndarray: A 2D numpy array with labels for arousal and valence.
        """
        if (self._labels_arousal is not None and
                self._labels_valence is not None):
            return np.column_stack(
                (self._labels_arousal, self._labels_valence))
        else:
            raise ValueError("Data was fetched before it was loaded. " +
                             "Do load_dataset() before calling get_dataset().")

    def delete_data(self):
        """
        Deletes the class members.
        """
        self._data = None
        self._labels_arousal = None
        self._labels_valence = None
        self._found_inds = []


def load_dataset(name,
                 npy_path,
                 path_arousal, path_valence,
                 path_arousal_std, path_valence_std,
                 remove_unlabeled_data_=True):
    """
    Loads the songs with IDs in 'songs' found in the paths
    into a dictionary with key='name'.

    Args:
        name (str): The desired name of the dataset.
        npy_path (pathlib.Path): A path leading to the
            folder with the features in .NPY-format.
        path_arousal (pathlib.Path): A path leading to the
            file with the arousal-values in .CSV-format.
        path_valence (pathlib.Path): A path leading to the
            file with the valence-values in .CSV-format.
        path_arousal_std (pathlib.Path): A path leading to the
            file with the arousal-std-values in .CSV-format.
        path_valence_std (pathlib.Path): A path leading to the
            file with the valence-std-values in .CSV-format.
        remove_unlabeled_data_ (bool): Run function 'remove_unlabeled_data' to
            remove unlabeled data. Default: True.
    """
    global _datasets

    # Load in data features.
    data, found_song_ids = load_features(npy_path)

    # Load annotations from corresponding .CSV files.
    arousal, valence = load_annotations(path_arousal, path_valence)
    arousal_std, valence_std = load_annotations(
        path_arousal_std, path_valence_std)

    # Remove song ids from 'found_song_ids'
    data, found_song_ids = remove_high_std_songs_from(data,
                                                      found_song_ids,
                                                      arousal_std,
                                                      valence_std)
    # Extract labels corresponding to 'found_song_ids'.
    labels_arousal, labels_valence = extract_samples(
        arousal, valence, found_song_ids)

    # Flatten data from 2D to 1D.
    labels_arousal, labels_valence = flatten_labels(
        labels_arousal, labels_valence)

    if remove_unlabeled_data_:
        data = remove_unlabeled_data(data)

    # Create and store the dataset.
    dataset = Dataset(data, labels_arousal, labels_valence, found_song_ids)
    _datasets[name] = dataset


def get_dataset(name):
    """
    Fetches the dataset with key=name.

    Args:
        name (str): The name of the dataset to be fetched.

    Raises:
        ValueError: Raised if dataset does not exist in storage.
    """
    if (name in _datasets):
        return(_datasets[name])
    else:
        s = f"Data set '{name}' not found. Call load_dataset() first."
        raise ValueError(s)


def delete_dataset(name):
    """
    Removes the dataset with key='name' from storage.

    Args:
        name (str): Name of the dataset to be removed.
    """
    global _datasets
    if (name in _datasets):
        _datasets[name].delete_data()
        del _datasets[name]
    else:
        warnings.warn(
            f"No dataset with key='{name}' found. Nothing was deleted.")
