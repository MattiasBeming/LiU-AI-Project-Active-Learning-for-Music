import numpy as np  # Version 1.19.1
import pandas as pd
from pathlib import Path

NO_OF_INITIAL_UNUSED_SAMPLES = 30


def load_features(npy_path):
    """
    Loads and formats feature data.

    Args:
        npy_path (pathlib.Path): A path to the features in .NPY format.

    Returns:
        data, found_song_ids (pandas.DataFrame, np.array): data is the
            feature data.
            found_song_ids stores what unique song_ids that the dataset
            will include.
    """
    data = pd.DataFrame(np.load(npy_path))
    found_song_ids = data.iloc[:, 0]
    index = pd.MultiIndex.from_arrays(
        [found_song_ids, data.iloc[:, 1]], names=["song_id", "sample"])
    data = data.iloc[:, 2:]
    data.index = index

    return data, np.unique(found_song_ids)


def load_annotations(path_arousal, path_valence):
    """
    Load in the annotations from respective CSV-files.

    Args:
        path_arousal (pathlib.Path): A path to arousal-labels.
        path_valence (pathlib.Path): A path to valence-labels.

    Raises:
        ValueError: Raised if any path if not found or not in .CSV format.

    Returns:
        arousal, valence (pandas.DataFrame, pandas.DataFrame): 2D pandas
            DataFrames with labels for arousal and valence repectively.
    """
    try:
        arousal = pd.read_csv(path_arousal, delimiter=",")
        valence = pd.read_csv(path_valence,  delimiter=",")
        return arousal, valence
    except (FileNotFoundError, UnicodeDecodeError):
        raise ValueError("Incorrect path to arousal or valence files.")


def extract_samples(arousal, valence, found_song_ids):
    """
    Extracts the samples for the found songs and then removes 'song_id' column.

    Args:
        arousal (pandas.DataFrame): 2D Pandas DataFram with labels for arousal.
            Dimensions: [song_id, sample]
        valence (pandas.DataFrame): 2D Pandas DataFram with values for valence.
            Dimensions: [song_id, sample]
        found_song_ids (np.array): A list of song

    Returns:
        arousal, valence (pandas.DataFrame, pandas.DataFrame):
            Modified arousal and valence 2D pandas objects.
    """
    # Extract the labels for the found data and remove the 'song_id'-column
    labels_arousal = arousal[arousal['song_id'].isin(
        found_song_ids)].set_index('song_id', drop=True)
    labels_valence = valence[valence['song_id'].isin(
        found_song_ids)].set_index('song_id', drop=True)
    return labels_arousal, labels_valence


def flatten_labels(labels_arousal, labels_valence):
    """
    Flattens the labels from 2D [song_id, sample] to 1D [song_id*sample].

    Args:
        labels_arousal (pandas.DataFrame): 2D Pandas DataFrame with labels for
            arousal. Dimensions: [song_id, sample]
        labels_valence (pandas.DataFrame): 2D Pandas DataFrame with values for
            valence. Dimensions: [song_id, sample]

    Returns:
        arousal_labels, valence_labels (pandas.DataFrame, pandas.DataFrame):
            Flattened labels in 1D pandas dataframe objects.
    """
    # Create MultiIndex
    samps = range(NO_OF_INITIAL_UNUSED_SAMPLES,
                  NO_OF_INITIAL_UNUSED_SAMPLES+len(labels_arousal.iloc[0, :]))
    foo = labels_arousal.index.values
    index = pd.MultiIndex.from_product([foo, samps],
                                       names=["song_id", "sample"])
    # Flatten to 1D numpy arraym then back to DataFrame with multilevel index.
    y_arousal = pd.DataFrame(labels_arousal.to_numpy().flatten(), index)
    y_valence = pd.DataFrame(labels_valence.to_numpy().flatten(), index)
    return y_arousal, y_valence


def remove_unlabeled_data(data):
    """
    Removes the first n samples from each song, where
        n = NO_OF_INITIAL_UNUSED_SAMPLES.

    Args:
        data (pandas.DataFrame): A dataframe containing features.
    """
    foo = list(range(0, NO_OF_INITIAL_UNUSED_SAMPLES))
    return data.drop(labels=foo, axis=0, level=1)


def csv_to_npy(csv_path, npy_path):
    """
    Converts the features-file from .CSV to .NPY.

    Args:
        csv_path (pathlib.Path): A path leading to the the features in
            .CSV-format.
        npy_path (pathlib.Path): A path leading to where the
            features are to be placed. (.NPY-format)
    """
    data = pd.read_csv(csv_path, index_col=[0], header=[0, 1, 2])
    np.save(npy_path, data)


def remove_high_std_songs_from(data, found_song_ids, arousal_std, valence_std):
    """
    Removes songs with high standard deviation from data and found_song_ids by
    looking at the *_cont_std.csv files.

    Args:
        data (pandas.DataFrame): A dataframe containing features.
        found_song_ids (np.array): A list of song ids.
        arousal_std (pandas.DataFrame): A dataframe of standard deviations
            for arousal.
        valence_std (pandas.DataFrame): A dataframe of standard deviations
            for valence.

    Returns:
        data, found_song_ids (pandas.DataFrame, np.array): A modified dataframe
        containing features and
        a modified list of song ids.
    """
    def _remove_high_std_songs_helper(found_song_ids, stds):
        song_id_list = []
        mean_list = []
        bad_songs = []

        # Calculate the mean for each row in stds
        # Saves the found song_ids
        # Iterate rows
        for i in range(stds.shape[0]):
            temp = []
            # Iterate cols
            for name, values in stds.iteritems():
                if name == "song_id":
                    song_id_list.append(values[i])
                else:
                    temp.append(values[i])

            mean_list.append(np.mean(temp))

        # Approximate normal distribution
        # Upper 99% probability interval
        threshold = np.mean(mean_list) + 2.576*np.std(mean_list)

        bad_songs = [song_id_list[i] for i, m in
                     enumerate(mean_list) if m > threshold]

        found_song_ids = np.setdiff1d(np.array(found_song_ids), bad_songs)
        return found_song_ids

    # Run helper function for both arousal and valence
    # to remove bad song_ids
    found_song_ids_new = _remove_high_std_songs_helper(
        found_song_ids, arousal_std)
    found_song_ids_new = _remove_high_std_songs_helper(
        found_song_ids_new, valence_std)

    print((f"Found {len(found_song_ids)-len(found_song_ids_new)} songs to "
           "be removed due to high standard deviation."))

    # Drop songs with high standard deviation not found in 'found_song_ids_new'
    data = data[data.index.get_level_values(
        'song_id').isin(found_song_ids_new)]

    return data, found_song_ids_new
