from threading import Thread
from api.storage import Dataset
from learning_profile import LearningProfile
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np


def _chunk_it(li: list, n_chunks: int):
    rest = len(li) % n_chunks
    chunk_size = int(np.floor(len(li) / n_chunks))
    for i in range(0, n_chunks):
        yield li[
            (i * chunk_size + min(rest, i)):
            ((i+1) * chunk_size + min(rest, i+1))
        ]


def _evaluate_learning_profiles(thread_id: int, lps: list, num_iterations: int,
                                seed_percent: float):
    for lp in tqdm(lps, position=thread_id*2,
                   desc=f"Thread {thread_id}", leave=False):

        # Print info
        tqdm.get_lock().acquire()
        tqdm.write(f"Thread {thread_id} > Evaluating {lp.get_id()}:")
        tqdm.write("{")

        tqdm.write("\tRunning until full training dataset has "
                   "been added to seed." if num_iterations == -1 else
                   f"\tRunning for {num_iterations} requested iterations.")

        train_data = lp.get_train_dataset().get_data()
        test_data = lp.get_test_dataset().get_data()

        n_samples_per_song = train_data.loc[
            int(train_data.index.get_level_values(0)[0]), :
        ].shape[0]

        n_train_songs = int(np.round(train_data.shape[0] / n_samples_per_song))
        n_test_songs = int(np.round(test_data.shape[0] / n_samples_per_song))

        tqdm.write(f"\tStarting with "
                   f"{int(np.ceil(n_train_songs * seed_percent))} "
                   f"({seed_percent*100:.1f}%) seed songs.")

        tqdm.write(f"\tNumber of samples per song: {n_samples_per_song}")

        tqdm.write(f"\tTraining dataset: {n_train_songs} songs "
                   f"({train_data.shape[0]} samples) @ "
                   f"{lp.get_train_dataset_name()}")

        tqdm.write(f"\tTesting dataset: {n_test_songs} songs "
                   f"({test_data.shape[0]} samples) @ "
                   f"{lp.get_test_dataset_name()}")

        tqdm.write(f"\tBatch size: {lp.get_batch_size()}")

        tqdm.write("}")
        tqdm.get_lock().release()

        # Evaluate current learning profile
        MSE_arousal, MSE_valence = _evaluate(
            lp, num_iterations, seed_percent, thread_id)
        lp.set_MSE(MSE_arousal, MSE_valence)

        # Print results
        tqdm.get_lock().acquire()
        tqdm.write(f"Thread {thread_id} > Finished with {lp.get_id()}:")
        tqdm.write("{")

        tqdm.write(f"\tArousal MSE per iteration: {lp.get_MSE_arousal()}")
        tqdm.write(f"\tValence MSE per iteration: {lp.get_MSE_valence()}")

        tqdm.write("}")
        tqdm.get_lock().release()

    tqdm.write(f"Thread {thread_id} finished successfully.")


def viability_phase(learning_profiles: list, num_iterations: int = -1,
                    seed_percent: float = 0.1, n_threads: int = 1):
    """
    Performs active learning for all Learning Profiles. However, instead of
    querying the user, pre-labeled data will be used for selected samples.

    The `learning_profiles` will be updated with MSE values.

    Args:
        learning_profiles (list): A list of all Learning Profiles to train.
        num_iterations (int): Number of evaluation iterations. Set to -1 if
            all training songs should be depleted. Default is -1.
        seed_percent (float): Percent of data to be used as seed.
            Default is 0.1.
        n_threads (int): The number of thread to use for evaluation.
            Default is 1.
    """

    # Print thread configuration
    print(f"Using {n_threads} threads for "
          f"{len(learning_profiles)} learning profiles.")

    threads = []

    # Start threads
    for i, lps in enumerate(_chunk_it(learning_profiles, n_threads)):

        # Print chunk info
        tqdm.get_lock().acquire()

        tqdm.write(f"Starting evaluation of {len(lps)} "
                   f"learning profiles on thread {i}:")

        tqdm.write("{")
        for lp in lps:
            tqdm.write(f"\t{str(lp)} | ID: {lp.get_id()}")
        tqdm.write("}")

        tqdm.get_lock().release()

        # Create and start thread
        th = Thread(target=_evaluate_learning_profiles,
                    args=[i, lps, num_iterations, seed_percent])
        th.start()
        threads.append(th)

    # Wait for threads to exit
    for th in threads:
        th.join()

    tqdm.write("All threads finished successfully!")


def _seed_song_ids(full_train_ds, seed_percent):
    unique_song_ids = np.unique(full_train_ds.get_contained_song_ids())
    nr_tr = int(np.ceil(len(unique_song_ids)*seed_percent))
    np.random.seed(42069)
    return list(np.random.choice(unique_song_ids, nr_tr, replace=False))


def _evaluate(lp: LearningProfile, num_iterations: int,
              seed_percent: float, thread_id: int):
    """
    Performs the following steps for the given learning profile:
        1: Add first song from training dataset to a seed dataset.
        2: Perform ML on this song to predict arousal/valence.
        3: Perform AL to find novel songs in training dataset to
            add to the seed dataset.
        4: Add these songs to the seed dataset.
        5: Fit ML model to the seed dataset and predict on validation dataset
            to get MSE values for current iteration.
        6: Repeat 3-5 for a specified number of iterations: `num_iterations`.

    Args:
        lp (LearningProfile): Learning profile to evaluate.
        num_iterations (int): Number of evaluation iterations. Set to -1 if
            all training songs should be depleted.
        seed_percent (float): Percent of data to be used as seed.
            Default is 0.1.
        thread_id (int): The ID of the calling thread.

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

    # Initialize training songs with first song id
    # Seed, percent
    seed_song_ids = _seed_song_ids(full_train_ds, seed_percent)

    # Mark all but the first song as unlabeled
    unlabeled_train_song_ids = list(
        set(full_train_song_ids).difference(seed_song_ids))

    # Get dataframe and labels of full training data
    full_train_dat = full_train_ds.get_data()
    full_train_ar = full_train_ds.get_arousal_labels()
    full_train_va = full_train_ds.get_valence_labels()

    # Derive number of samples per song (assume same for all songs)
    n_samples_per_song = full_train_dat.loc[seed_song_ids[0], :].shape[0]

    # Get current unlabeled features
    unlabeled_features = full_train_dat.loc[unlabeled_train_song_ids, :]

    # Construct seed dataset
    seed_ds = Dataset(
        data=full_train_dat.loc[seed_song_ids, :],
        labels_arousal=full_train_ar.loc[seed_song_ids, :],
        labels_valence=full_train_va.loc[seed_song_ids, :]
    )

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
    # Step 3-5 #
    ############

    for _ in tqdm(range(num_iterations), position=thread_id * 2 + 1,
                  desc="    Current LP", leave=False):

        # Perform Active Learning
        songs_to_add = al_func(
            seed_ds, Dataset(unlabeled_features),
            pred_unlabeled, lp.get_batch_size(),
            n_samples_per_song, model
        )

        # Add selected songs to seed dataset
        seed_ds.add_datapoints(
            features=full_train_dat.loc[songs_to_add, :],
            arousal_labels=full_train_ar.loc[songs_to_add, :],
            valence_labels=full_train_va.loc[songs_to_add, :]
        )
        seed_song_ids += songs_to_add

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
