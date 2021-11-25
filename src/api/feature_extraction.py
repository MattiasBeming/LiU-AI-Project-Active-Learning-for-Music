# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst,
# Xavier Bresson, EPFL LTS2.

# All features are extracted
# using [librosa](https://github.com/librosa/librosa).

# Note:
# This file was edited to work for emo-music in our project.
# All credit for the core implementation is given to the original authors.

# OBS, RUN THIS FILE FROM SRC FOLDER
import os
import multiprocessing
import warnings
import numpy as np
from numpy.core.defchararray import count
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
import psutil
import time


def get_audio_path(audio_dir, song_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the Song ID.

    Args:
        audio_dir (String): The directory where the audio is stored.
        song_id (String): The Song ID.

    Returns:
        String: The path to the mp3 file with the given song ID.
    """
    return Path(audio_dir) / (str(song_id) + '.mp3')


def load(filepath):
    """
    Load in metadata from filepath.

    Args:
        filepath (string/path): path to file.

    Returns:
        pd.DataFrame: dataframe containing metadata.
    """
    tracks = pd.read_csv(filepath, index_col=0, header=[0])

    # Format the data.
    # Remove "tabs" from strings etc.
    tracks["file_name"] = tracks["file_name"].map(lambda s: s.strip())
    tracks["Artist"] = tracks["Artist"].map(lambda s: s.strip())
    tracks["Song title"] = tracks["Song title"].map(lambda s: s.strip())

    tracks["start of the segment (min.sec)"] = \
        tracks["start of the segment (min.sec)"].map(lambda s: float(s))
    tracks["end of the segment (min.sec)"] = \
        tracks["end of the segment (min.sec)"].map(lambda s: float(s))

    tracks["Genre"] = tracks["Genre"].map(lambda s: s.strip())
    tracks["Genre"] = tracks["Genre"].astype('category')

    return tracks


def save_npy(song_id):
    """
    Load song with song_id into memory and convert it to a list of values
    usable by python. Store this list in a new file (.npy).

    Args:
        song_id (int): The song ID.

    Returns:
        Tuple(int, float): Returns a tuple with song_id, sample rate.
    """
    try:
        filedir = Path('data/emo-music-features/clips_45sec/clips_45seconds')
        filepath = get_audio_path(filedir, song_id)
        sound = AudioSegment.from_file(filepath)
        samples = sound.get_array_of_samples()
        samples = np.array(samples)

        dir_path = Path('data/samples')
        if not dir_path.is_dir():
            # Create dir
            dir_path.mkdir()

        filepath = dir_path / (str(song_id) + '.npy')
        np.save(filepath, samples)
    except Exception as e:
        print("Removing invalid song id: ", song_id)
        print(repr(e))
        return (None, None)

    return (song_id, sound.frame_rate)


def load_npy(song_id, sample_id, n_samples):
    """
    Load song from .npy files and split it accordingly.

    Args:
        song_id (int): -
        sample_id (int): -
        n_samples (int): -

    Returns:
        float[]: returns the part of samples for song_id that
                 corresponds to the given sample_id.
    """
    filepath = Path('data/samples') / (str(song_id) + '.npy')
    samples = np.load(filepath)
    split_samples = np.array_split(samples, n_samples)
    return split_samples[sample_id]


def columns():
    """
    Constructs columns for the feature dataframe.

    Returns:
        pd.MultiIndex: -
    """
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rms=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(zip_):
    """
    Computes the features for a single sample/row.

    Args:
        zip_ (tuple): Contains unique_id, song_id, song_ids,
                      n_samples, sample rate.

    Returns:
        pd.Series: A row of calculated features,
        one row == one sample of 500 ms.
    """
    unique_id = zip_[0]
    song_id = zip_[1]
    song_ids = zip_[2]
    n_samples = zip_[3]
    song_idx = list(song_ids).index(song_id)
    sample_id = unique_id - n_samples*song_idx
    sound_sr = zip_[4]

    features = pd.Series(index=columns(), dtype=np.float32, name=unique_id)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        # Calculate all the statistics for the feature "name".
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        samples = load_npy(song_id, sample_id, n_samples)

        x = np.array(samples).astype(np.float32)/32768  # 16 bit
        sr = 22050
        x = librosa.core.resample(
            x, sound_sr, sr, res_type='kaiser_best')

        # Get all features
        f = librosa.feature.zero_crossing_rate(
            x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None, fmin=65.41))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)

        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)

        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        n_fft = 2048//16
        hop_length = n_fft // 4
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        assert stft.shape[0] == 1 + n_fft // 2
        assert np.ceil(
            len(x)/hop_length) <= stft.shape[1] <= np.ceil(len(x)/hop_length)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(
            S=stft, frame_length=n_fft, hop_length=hop_length)
        feature_stats('rms', f)

        f = librosa.feature.spectral_centroid(
            S=stft, n_fft=n_fft, hop_length=hop_length)
        feature_stats('spectral_centroid', f)

        f = librosa.feature.spectral_bandwidth(
            S=stft, n_fft=n_fft, hop_length=hop_length)
        feature_stats('spectral_bandwidth', f)

        f = librosa.feature.spectral_contrast(
            S=stft, n_bands=6, n_fft=n_fft, hop_length=hop_length)
        feature_stats('spectral_contrast', f)

        f = librosa.feature.spectral_rolloff(
            S=stft, n_fft=n_fft, hop_length=hop_length)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(
            sr=sr, S=stft**2, n_fft=n_fft, hop_length=hop_length,
            fmax=sr//2, n_mels=32)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('-> Song_id: {}, Sample_id: {} Warning: {}\n'.format(
            song_id, sample_id, repr(e)))
    return features


def remove_songs_with_missing_data(features, n_samples):
    """
    Removes songs with missing data from a copy of the features dataframe.

    Args:
        features (pd.DataFrame): dataframe with features.
        n_samples (int): Number of samples.

    Returns:
        pd.DataFrame: The new dataframe.
    """
    song_id = [i for i in np.array(features["song_id"])]
    true_unique = [i for i in range(0, n_samples * len(np.unique(song_id)))]

    # Get a list of all faulty song ids
    faulty_song_ids = []
    offset = 0
    count = 0
    for i in np.array(features.index):
        i_ = int(i) - offset
        if i_ != true_unique[count]:
            offset += abs(true_unique[count] - i_)
            faulty_song_ids.append(song_id[count - offset])
        count += 1
    faulty_song_ids = np.unique(np.array(faulty_song_ids))

    n_of_songs_to_remove = len(faulty_song_ids)
    print(f"Removing {n_of_songs_to_remove} songs...")

    # Get a list of all row ids to drop
    ids_to_drop = []
    for id_ in faulty_song_ids:
        ids = list(features[(features["song_id"] == id_)].index)
        ids_to_drop += ids

    # Remove songs
    features_new = features.drop(ids_to_drop)

    # Re-assign indices
    new_unique_ids = [i for i in range(0, features_new.shape[0])]
    features_new.index = new_unique_ids
    print(("Removed the following song(s) with id(s): "
           f"{faulty_song_ids} (⌐□_□)"))
    return features_new


def save(features, ndigits):
    features.to_csv('data/features_librosa.csv',
                    float_format='%.{}e'.format(ndigits))


def test(features, n_samples):
    """
    Tests dataframe such that all sections of
    songs each contain n_samples samples.

    I.e. if missing rows can't be interpolated/extrapolated
    the entire song needs to be removed. And this should be done in an earlier
    step (the function: remove_songs_with_missing_data()).

    Args:
        features (pd.dataframe): feature dataframe.
        n_samples (int): Number of samples in one song.
    """
    try:
        count_samples = 0
        count_unique_id = 0
        for sample_id in np.array(features['sample_id']):
            assert(sample_id == count_samples)

            count_samples += 1
            count_unique_id += 1
            if count_samples == n_samples:
                count_samples = 0
    except Exception as e:
        print((f"sample with id: {sample_id} and unique_id: "
               f"{count_unique_id} has failed the assertion."
               f" Exception: {repr(e)}"))
    return


def main():
    """
    Calculates features given mp3 songs and metadata, saves result as csv-file.
    """
    ###########################
    ### Load/transform data ###
    ###########################

    start_time = time.time()
    os.chdir('./../')  # Change to parent directory
    filename = Path('data/emo-music-features/annotations/songs_info.csv')
    tracks = load(filename)

    n_samples = 91  # Number of samples (91 -> 0.5 seconds per sample)

    # Limit for number of missing samples in a row
    # before removing song instead of interpolating / extrapolating
    limit_interpolate = 2
    limit_extrapolate = 1

    # If crash -> lower this value!
    nb_workers = psutil.cpu_count(logical=False)
    print(f'Working with {nb_workers} processes.')

    # Create a pool of workers
    pool = multiprocessing.Pool(nb_workers)

    print(("Create and save sample files (.npy) from"
           " mp3 files, using AudioSegment..."))

    # Retrieve song ids and samples rates
    sound_fr_it = pool.imap(save_npy, tracks.index)
    song_ids, sound_fr = map(list, zip(*sound_fr_it))
    song_ids = np.array([int(i) for i in song_ids if i])

    sound_fr = np.array([int(i) for i in sound_fr if i])
    sound_fr_rep = np.repeat(sound_fr, n_samples)

    song_ids_rep = np.repeat(song_ids, n_samples)
    unique_ids = [i for i in range(n_samples * len(song_ids))]

    # Construct list of tuples
    tids_repeat = zip(
        unique_ids, song_ids_rep, [song_ids]*len(song_ids_rep),
        [n_samples]*len(song_ids_rep), sound_fr_rep)

    ############################
    ### Extract feature data ###
    ############################

    # Extract features from saved sample files (.npy)
    print("Starting to generate features...")
    it = pool.imap_unordered(compute_features, tids_repeat)

    # Create features dataframe
    features = pd.DataFrame(
        index=unique_ids, columns=columns(), dtype=np.float32)

    count = 0
    # Add extracted features to DataFrame (features)
    for i, row in enumerate(tqdm(it, total=len(unique_ids))):
        if not pd.isnull(row).values.any():
            features.loc[row.name] = row
        else:
            count += 1

        if i % 10000 == 0:
            save(features, 10)

    print(f"{count} NaN row(s) were found.")

    # Insert a column for sample ids and song ids (order matters)
    features.insert(0, 'sample_id', list(range(n_samples)) * len(song_ids))
    features.insert(0, 'song_id', song_ids_rep)

    print("All features generated!")

    ##################
    ### Clean data ###
    ##################

    print("Interpolate/Extrapolate missing values...")

    # Interpolate/Extrapolate missing rows in chunks of n_samples
    # in order to avoid overlapping songs
    n_songs = features.shape[0] // n_samples
    for song in range(n_songs):
        start = song * n_samples
        end = start + n_samples

        # Interpolate
        features.iloc[start:end] = features.iloc[start:end].interpolate(
            axis=0, limit=limit_interpolate, limit_area="inside",
            limit_direction="both")

        # Extrapolate
        features.iloc[start:end] = features.iloc[start:end].interpolate(
            axis=0, limit=limit_extrapolate, limit_area="outside",
            limit_direction="both")

    # Drop all NaN rows
    features = features.dropna()

    # Remove songs that can't be interpolated/extrapolated
    features = remove_songs_with_missing_data(features, n_samples)

    ###################
    ### Test & Save ###
    ###################

    test(features, n_samples)
    save(features, 10)

    print("Total Time: ", time.time() - start_time)


if __name__ == "__main__":
    main()
