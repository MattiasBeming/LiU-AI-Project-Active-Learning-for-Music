from sklearn.metrics import mean_squared_error
from learning_profile import LearningProfile
from phase_utils import poll_seed_song_ids
from api.storage import Dataset
from api import gui
from pathlib import Path
import numpy as np
from tqdm import tqdm

from api.windows import QUERY_MODE_AROUSAL, QUERY_MODE_VALENCE


def user_phase(learning_profiles: list, data_dir: Path,
               num_iterations: int = -1, seed_percent: float = 0.1,
               audio_file_ext: str = "mp3"):
    """
    Performs active learning for all Learning Profiles. The user will be
    queried for labels.

    Args:
        learning_profiles (list): The Learning Profiles to perform AL on.
        data_dir (pathlib.Path): Should point to a directory with audio files
            to play. The files should be named `[song_id].[audio_file_ext]`
            (e.g. `69.mp3`).
        num_iterations (int, optional): Number of evaluation iterations. Set
            to -1 if all training songs should be depleted. Defaults to -1.
        seed_percent (float, optional):  Percent of training data to be used
            as seed. Defaults to 0.1.
        audio_file_ext (str, optional): The file extension of the audio files
            in `data_dir`. Defaults to "mp3".

    Raises:
        FileNotFoundError: If `data_dir` does not point to a valid directory.

    Yields:
        LearningProfile: For all `learning_profiles` one by one as
            they are completed.
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Invalid directory: {data_dir.absolute()}")

    for lp in learning_profiles:
        print(f"Evaluating:\n\t{lp}")
        MSE_arousal, MSE_valence = _evaluate(lp, data_dir, num_iterations,
                                             seed_percent, audio_file_ext)
        lp.set_MSE(MSE_arousal, MSE_valence)

        print(f"Done!")
        print(f"Arousal MSE per iteration: {lp.get_MSE_arousal()}")
        print(f"Valence MSE per iteration: {lp.get_MSE_valence()}")

        yield lp


def _evaluate(lp: LearningProfile, data_dir: Path,
              num_iterations: int, seed_percent: float,
              audio_file_ext: str = "mp3"):
    """
    Performs the following steps for the given learning profile:
        1: Add first songs from training dataset to a seed dataset.
        2: Perform ML on these songs to predict arousal/valence.
        3: Perform AL to find novel songs in training dataset.
        4: Query the user for arousal/valence labels for these novel songs.
        5: Add these songs with queried labels to the seed dataset.
        6: Fit ML model to the seed dataset and predict on validation dataset
            to get MSE values for current iteration.
        7: Repeat 3-6 for a specified number of iterations: `num_iterations`.

    Args:
        lp (LearningProfile): Learning profile to evaluate.
        data_dir (pathlib.Path): Should point to a directory with audio files
            to play. The files should be named `[song_id].[audio_file_ext]`
            (e.g. `69.mp3`).
        num_iterations (int): Number of evaluation iterations. Set to -1 if
            all training songs should be depleted.
        seed_percent (float): Percent of training data to be used as seed.
        audio_file_ext (str): The file extension of the audio files
            in `data_dir`.

    Returns:
        (np.ndarray, np.ndarray): Arrays with arousal and valence MSE values.
    """

    ########
    # Init #
    ########

    # Get functions
    al_func = lp.get_al_function()
    ml_func = lp.get_ml_function()

    # Get full datasets
    full_train_ds = lp.get_train_dataset()
    full_validation_ds = lp.get_test_dataset()

    # Get song ids of full training data
    full_train_song_ids = full_train_ds.get_contained_song_ids()

    ##########
    # Step 1 #
    ##########

    # Initialize training songs with first seed songs
    seed_song_ids = poll_seed_song_ids(full_train_ds, seed_percent)

    # Mark all but the first seed songs as unlabeled
    unlabeled_train_song_ids = list(
        set(full_train_song_ids).difference(seed_song_ids))

    # Get dataframe of full training data
    full_train_dat = full_train_ds.get_data()

    # Derive number of samples per song (assume same for all songs)
    n_samples_per_song = full_train_dat.loc[seed_song_ids[0], :].shape[0]

    # Get current unlabeled features
    unlabeled_features = full_train_dat.loc[unlabeled_train_song_ids, :]

    # Construct seed dataset
    seed_ds = Dataset(
        data=full_train_dat.loc[seed_song_ids, :],
        labels_arousal=full_train_ds.get_arousal_labels(
        ).loc[seed_song_ids, :],
        labels_valence=full_train_ds.get_valence_labels(
        ).loc[seed_song_ids, :]
    )

    seed_ds.sliding_window_inherit(full_train_ds, alter=False)

    # Init arrays for storing MSE
    MSE_arousal_arr = []
    MSE_valence_arr = []

    ##########
    # Step 2 #
    ##########

    # Run ML an initial time to get base predictions
    model = ml_func(seed_ds, lp.get_hyper_parameters(), train=True)
    pred_unlabeled = model.predict(unlabeled_features)

    # Process all songs if requested
    if num_iterations == -1:
        num_iterations = int(np.ceil(
            len(lp.get_train_dataset().get_contained_song_ids()) /
            lp.get_batch_size()
        ))

    ############
    # Step 3-6 #
    ############

    for _ in tqdm(range(num_iterations), desc="LP Progress"):

        # Perform Active Learning
        songs_to_add = al_func(
            seed_ds, Dataset(unlabeled_features),
            pred_unlabeled, lp.get_batch_size(),
            n_samples_per_song, model
        )

        # Query user for labels
        for sID in songs_to_add:

            # Derive file path from song ID
            p = data_dir.joinpath(Path(f"{sID}.{audio_file_ext}"))

            # Query arousal
            ar_ref = gui.query_dynamic(p, QUERY_MODE_AROUSAL)
            ar_res = gui.wait_result(ar_ref)

            # Query valence
            va_ref = gui.query_dynamic(p, QUERY_MODE_VALENCE)
            va_res = gui.wait_result(va_ref)

            # Add song to seed dataset
            features_to_add = full_train_ds.get_raw_data().loc[[sID], :]
            seed_ds.add_songs(
                features_to_add,
                ar_res[features_to_add.index.get_level_values(1)
                       .astype(np.int64), 1],
                va_res[features_to_add.index.get_level_values(1)
                       .astype(np.int64), 1]
            )
            seed_song_ids.append(sID)

        # Update unlabeled meta data
        unlabeled_train_song_ids = list(
            set(full_train_song_ids).difference(seed_song_ids))
        unlabeled_features = full_train_dat.loc[unlabeled_train_song_ids, :]

        # Get validation_labels dataset labels
        validation_labels = full_validation_ds.get_labels()

        # Perform ML
        model = ml_func(seed_ds, lp.get_hyper_parameters(), train=True)
        y_pred = model.predict(full_validation_ds.get_data())
        MSE_arousal = mean_squared_error(
            validation_labels.iloc[:, 0], y_pred[:, 0])
        MSE_valence = mean_squared_error(
            validation_labels.iloc[:, 1], y_pred[:, 1])
        pred_unlabeled = model.predict(unlabeled_features)

        # Save result
        MSE_arousal_arr.append(MSE_arousal)
        MSE_valence_arr.append(MSE_valence)

    return MSE_arousal_arr, MSE_valence_arr
