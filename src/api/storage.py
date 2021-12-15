from api.transformer import *
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm


# Global values
NO_OF_INITIAL_UNUSED_SAMPLES = 30
_datasets = {}


class Dataset():
    def __init__(self, data=None, labels_arousal=None, labels_valence=None):
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
        """
        self._data = data
        self._labels_arousal = labels_arousal
        self._labels_valence = labels_valence
        self._sliding_window = 0
        self._throw_data = False
        self._prior = np.array([])

    def add_datapoints(self, features, arousal_labels, valence_labels):
        """
        Appends datapoints to this dataset. Note that if this dataset is
        configured with a sliding window, `features` must contain values
        for the window slots. See `get_data_shape()`.

        Args:
            features (pandas.DataFrame): 2D DataFrame with dimensions
                [[song_id, sample], features].
            arousal_labels (pandas.DataFrame): Single column DataFrame with
                [[song_id, sample], arousal].
            valence_labels (pandas.DataFrame): Single column DataFrame with
                [[song_id, sample], valence].
        """
        self._data = pd.concat([self._data, features], levels=2)
        self._labels_arousal = pd.concat(
            [self._labels_arousal, arousal_labels])
        self._labels_valence = pd.concat(
            [self._labels_valence, valence_labels])

    def add_songs(self, features, arousal_labels, valence_labels):
        """
        Add songs to dataset. Note that `features` should not
        contain any sliding window columns. These columns will be
        added by the function if the dataset is configured with a
        sliding window.

        Args:
            features (pandas.DataFrame): 2D DataFrame with dimensions
                [[song_id, sample], features].
            arousal_labels (np.ndarray): 2D numpy array of arousal values for
                            the songs in features. Should be same order
                            with regards to samples as in
                            the `features` parameter. Dimensions
                            should be (song_ids*samples, 1).
            valence_labels (np.ndarray): 2D numpy array of valence values for
                            the songs in features. Should be same order
                            with regards to samples and song ids as in
                            the `features` parameter. Dimensions
                            should be (song_ids*samples, 1).
        """
        # Create pandas dataframe for arousal and valence labels
        multi_index = features.index
        labels_arousal = pd.DataFrame(
            data=arousal_labels, index=multi_index)
        labels_valence = pd.DataFrame(
            data=valence_labels, index=multi_index)

        # Create temp dataset and add the data to it
        temp_ds = Dataset(
            data=features,
            labels_arousal=labels_arousal,
            labels_valence=labels_valence
        )

        # Call sliding window on temp dataset with stored params
        temp_ds.sliding_window_train(
            sliding_window=self._sliding_window, prior=self._prior)

        # Use temp dataset to add new datapoints to self
        self.add_datapoints(
            features=temp_ds.get_data(),
            arousal_labels=temp_ds.get_arousal_labels(),
            valence_labels=temp_ds.get_valence_labels()
        )

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
            if(self._throw_data):
                return self._data.drop(
                    labels=self._data.index.get_level_values(1)[
                        0:self._sliding_window
                    ], axis=0, level=1
                )
            else:
                return self._data
        else:
            raise ValueError("Data was fetched before it was loaded. " +
                             "Do load_dataset() before calling get_dataset().")

    def get_first_n_samples_of_each_song(self, n):
        """
        Slices data to keep only the first n samples of each song.
        Keeps the samples from 0 to n-1.

        Args:
            n (int): Number of samples to keep.
        """
        temp = slice(NO_OF_INITIAL_UNUSED_SAMPLES,
                     NO_OF_INITIAL_UNUSED_SAMPLES+n-1)
        return(self._data.loc[(slice(None), temp), :])

    def get_arousal_labels(self):
        """
        Returns the labels for arousal for this dataset.

        Raises:
            ValueError: If 'labels_arousal' is None.

        Returns:
            pandas.DataFrame: A 1D pandas dataframe with labels for arousal.
        """
        if (self._labels_arousal is not None):
            if(self._throw_data):
                return self._labels_arousal.drop(
                    labels=self._labels_arousal.index.get_level_values(1)[
                        0:self._sliding_window
                    ], axis=0, level=1
                )
            else:
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
            if(self._throw_data):
                return self._labels_valence.drop(
                    labels=self._labels_valence.index.get_level_values(1)[
                        0:self._sliding_window
                    ], axis=0, level=1
                )
            else:
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
            pd.DataFrame: A 2D DataFrame with labels for arousal and valence.
        """
        return pd.concat(
            [self.get_arousal_labels(), self.get_valence_labels()],
            axis=1
        )

    def get_contained_song_ids(self):
        """
        Returns the song_ids contained by this dataset.

        Returns:
            np.ndarray: 1D array with song_ids in the dataset.
        """
        return np.array(self._data.index.levels[0])

    def delete_data(self):
        """
        Deletes the class members.
        """
        self._data = None
        self._labels_arousal = None
        self._labels_valence = None
        self._sliding_window = None
        self._throw_data = None
        self._prior = None

    def sliding_window_train(self, sliding_window, prior=np.array([])):
        """
        Adds sliding window features to the data using target labels. Three
        intended use cases exist:
            * Send in sliding_window > 1 and a prior of length 1, such as
              np.array([0]). Then the prior used will an array of length
              sliding_window with only zeros.
            * Send in 2*sliding_window == len(prior), then the sent in prior
              will be used.
            * Send in sliding_window > 0 and empty prior, then the
              sliding_window first target from arousal and valence will be used
              as "priors". The corresponding data points will be "thrown away".

        Args:
            sliding_window (int): Number of previous samples to look at.
            prior (np.array, optional): Prior to use for first samples.
                    Defaults to np.array([]).
        """

        # Create column labels for sliding window
        col_labels = [
            f"sw_a{int(1 + i/2)}" if (i % 2 == 0)
            else f"sw_v{int(1 + (i-1)/2)}"
            for i in range(0, 2*sliding_window)
        ]

        if(self._sliding_window > 0):
            # Remove previous sliding window
            self._data.drop(
                columns=self._data.columns[-2*self._sliding_window:],
                inplace=True
            )

        # Set new sliding window
        self._sliding_window = sliding_window

        if(sliding_window > 0 and len(prior) > 0):
            # Don't throw away data
            self._throw_data = False

            # Create matrix with zeros to append to data and fill
            n, _ = self._data.shape
            zeros = pd.DataFrame(
                np.zeros((n, 2*sliding_window)), columns=col_labels
            )

            # Append matrix to data
            self._data = pd.concat(
                [self._data.reset_index(), zeros], axis=1
            )

            # Set index
            self._data.set_index(
                ['song_id', 'sample'],
                inplace=True
            )

            if(len(prior) == 1 and sliding_window > 1):
                # Use prior duplicated 2*sliding_window times
                prior = np.repeat(prior, 2*sliding_window)

            # Use prior
            for _, song in tqdm(self._data.groupby(level=0)):
                prev_sample = []
                prev_sample_index = 0

                for index, _ in song.iterrows():
                    if(len(prev_sample) == 0):
                        # Insert prior at first sample of each song
                        self._data.loc[index][col_labels] = prior
                    else:
                        # Use previous sample to shift window
                        self._data.loc[index][col_labels[2:]] = \
                            list(prev_sample.iloc[-2*sliding_window:-2])

                        # Use previous target to update first 2 positions
                        self._data.loc[index][col_labels[0:2]] = \
                            list(self._labels_arousal.loc[prev_sample_index]) \
                            + list(self._labels_valence.loc[prev_sample_index])

                    # Update prev_sample and prev_sample_index
                    prev_sample = self._data.loc[index]
                    prev_sample_index = index

        elif sliding_window == 0:
            # Don't throw away data
            self._throw_data = False

        else:
            # Case where prior is empty and sliding window > 0

            # Throw away data
            self._throw_data = True

            # Create matrix with zeros to append to data and fill
            n, _ = self._data.shape
            zeros = pd.DataFrame(
                np.zeros((n, 2*sliding_window)), columns=col_labels)

            # Append matrix to data
            self._data = pd.concat(
                [self._data.reset_index(), zeros], axis=1
            )

            # Set index
            self._data.set_index(['song_id', 'sample'], inplace=True)

            for _, song in tqdm(self._data.groupby(level=0)):
                prev_sample = []
                prev_sample_index = 0
                sample_num = 0
                prior = []

                for index, _ in song.iterrows():
                    if(sample_num == sliding_window):
                        # Insert prior at first sample of each song
                        self._data.loc[index][col_labels] = prior
                        prev_sample = self._data.loc[index]
                        prev_sample_index = index
                    elif(sample_num < sliding_window):
                        # Build prior
                        prior = list(self._labels_arousal.loc[index]) + \
                            list(self._labels_valence.loc[index]) + prior
                    else:
                        # Use previous sample to shift window
                        self._data.loc[index][col_labels[2:]] = \
                            list(prev_sample.iloc[-2 * sliding_window:-2])

                        # Use previous target to update first 2 positions
                        self._data.loc[index][col_labels[0:2]] = \
                            list(self._labels_arousal.loc[prev_sample_index]) \
                            + list(self._labels_valence.loc[prev_sample_index])

                    sample_num += 1

                    # Update prev_sample and prev_sample_index
                    prev_sample = self._data.loc[index]
                    prev_sample_index = index

        # Store prior
        self._prior = prior

    def sliding_window_test(self, regressor, sliding_window, prior):
        """
        Adds sliding window features to the data using a given regressor. Three
        intended use cases exist:
            * Send in sliding_window > 1 and a prior of length 1, such as
              np.array([0]). Then the prior used will an array of length
              sliding_window with only zeros.
            * Send in 2*sliding_window == len(prior), then the sent in prior
              will be used.
            * Send in sliding_window > 0 and an array of priors, one for each
              song in the data. The sliding_window first samples of each song
              will be "thrown away".

        Args:
            regressor (some sklearn regressor): Some sklearn regressor to use
            for predictions.
            sliding_window (int): Number of previous samples to look at.
            prior (np.array): Prior to use for first samples.
        """

        # Create column labels for sliding window
        col_labels = [
            f"sw_a{int(1 + i/2)}" if (i % 2 == 0)
            else f"sw_v{int(1 + (i-1)/2)}"
            for i in range(0, 2*sliding_window)
        ]

        if(self._sliding_window > 0):
            # Remove previous sliding window
            self._data.drop(
                columns=self._data.columns[-2*self._sliding_window:],
                inplace=True
            )

        if sliding_window == 0:
            # Don't throw away data
            self._throw_data = False
            return

        # Set new sliding window
        self._sliding_window = sliding_window

        # Create matrix with zeros to append to data and fill
        n, _ = self._data.shape
        zeros = pd.DataFrame(
            np.zeros((n, 2*sliding_window)), columns=col_labels
        )

        # Append matrix to data
        self._data = pd.concat(
            [self._data.reset_index(), zeros], axis=1
        )

        # Set index
        self._data.set_index(['song_id', 'sample'], inplace=True)

        if prior.ndim == 1:
            # Don't throw away data
            self._throw_data = False

            if(len(prior) == 1 and sliding_window > 1):
                # Use prior duplicated 2*sliding_window times
                prior = np.repeat(prior, 2*sliding_window)

            # Use prior
            for _, song in tqdm(self._data.groupby(level=0)):
                prev_sample = []

                for index, _ in song.iterrows():
                    if(len(prev_sample) == 0):
                        # Insert prior at first sample of each song
                        self._data.loc[index][col_labels] = prior
                    else:
                        # Use previous sample to shift window
                        self._data.loc[index][col_labels[2:]] = \
                            list(prev_sample.iloc[-2*sliding_window:-2])

                        # Use regressor to predict and update first 2 positions
                        self._data.loc[index][col_labels[0:2]] = \
                            regressor.predict([prev_sample])[0]

                    # Update prev_sample
                    prev_sample = self._data.loc[index]
        else:
            # Throw away data
            self._throw_data = True

            # Use different prior for each song
            song_num = 0

            for _, song in tqdm(self._data.groupby(level=0)):
                prev_sample = []
                song_prior = prior[song_num]
                sample_num = 0

                for index, _ in song.iterrows():
                    if(sample_num >= sliding_window):
                        if(sample_num == sliding_window):
                            # Insert prior
                            self._data.loc[index][col_labels] = song_prior
                        else:
                            # Use previous sample to shift window
                            self._data.loc[index][col_labels[2:]] = \
                                list(prev_sample.iloc[-2 * sliding_window:-2])

                            # Use regressor to predict and update first 2 pos
                            self._data.loc[index][col_labels[0:2]] = \
                                regressor.predict([prev_sample])[0]

                    # Update prev_sample and sample number
                    prev_sample = self._data.loc[index]
                    sample_num += 1
                song_num += 1

    def get_data_shape(self):
        return self.get_data().shape

    def get_sliding_window_size(self):
        return self._sliding_window


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
    dataset = Dataset(data, labels_arousal, labels_valence)
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
