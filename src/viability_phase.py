from api.storage import Dataset
from learning_profile import LearningProfile
from pathlib import Path
from api import storage
from api import ml
from api import al
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np


def viability_phase(learning_profiles: list, num_iterations: int = 1,
                    seed_percent: float = 0.1):
    """
    Performs active learning for all Learning Profiles. However, instead of
    querying the user, pre-labeled data will be used for selected samples.

    Args:
        learning_profiles (list): A list of all Learning Profiles to train.
        num_iterations (int): Number of evaluation iterations. Set to -1 if
            all training songs should be depleted. Default is -1.
        seed_percent (float): Percent of data to be used as seed. Default is 0.1.
    """

    # Perform viability test for all learning profiles
    for lp in learning_profiles:
        print(f"Evaluating {lp}")
        MSE_arousal, MSE_valence = _evaluate(lp, num_iterations, seed_percent)
        lp.set_MSE(MSE_arousal, MSE_valence)

        print(f"Done!")
        print(f"Arousal MSE per iteration: {lp.get_MSE_arousal()}")
        print(f"Valence MSE per iteration: {lp.get_MSE_valence()}")


def _seed_song_ids(full_train_ds, seed_percent):
    unique_song_ids = np.unique(full_train_ds.get_contained_song_ids())
    nr_tr = int(np.ceil(len(unique_song_ids)*seed_percent))
    np.random.seed(42069)
    return list(np.random.choice(unique_song_ids, nr_tr, replace=False))


def _evaluate(lp: LearningProfile, num_iterations: int, seed_percent: float):
    """
    Performs the following steps for the given learning profile:
        1: Add first song from training dataset to a seed dataset.
        2: Perform ML on this song to predict arousal/valence.
        3: Perform AL to find novel songs in training dataset to
            add to the seed dataset.
        4: Add these songs to the seed dataset.
        5: Fit ML model to the seed dataset and predict on test dataset
            to get MSE values for current iteration.
        6: Repeat 3-5 for a specified number of iterations: `num_iterations`.

    Args:
        lp (LearningProfile): Learning profile to evaluate.
        num_iterations (int): Number of evaluation iterations. Set to -1 if
            all training songs should be depleted.
        seed_percent (float): Percent of data to be used as seed. Default is 0.1.
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
    full_test_ds = lp.get_test_dataset()

    # Get song ids of full training data
    full_train_song_ids = full_train_ds.get_contained_song_ids()

    ##########
    # Step 1 #
    ##########

    # Initialize training songs with first song id
    # Seed, percent
    seed_song_ids = _seed_song_ids(full_train_ds, seed_percent)
    print(
        f"Running with {len(seed_song_ids)} ({seed_percent*100}%) seed songs.")

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
        print("Running until full training dataset has been added to seed.")
    else:
        print(f"Running for {num_iterations} requested iterations.")

    print(f"Batch size: {lp.get_batch_size()}")

    ############
    # Step 3-5 #
    ############

    for _ in tqdm(range(num_iterations)):

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

        # Get test dataset labels
        test_labels = full_test_ds.get_labels()

        # Perform ML
        model = ml_func(seed_ds, lp.get_hyper_parameters(), train=True)
        y_pred = model.predict(full_test_ds.get_data())
        MSE_arousal = mean_squared_error(test_labels.iloc[:, 0], y_pred[:, 0])
        MSE_valence = mean_squared_error(test_labels.iloc[:, 1], y_pred[:, 1])
        pred_unlabeled = model.predict(unlabeled_features)

        # Save result
        MSE_arousal_arr.append(MSE_arousal)
        MSE_valence_arr.append(MSE_valence)

    return MSE_arousal_arr, MSE_valence_arr

# Mock init


BATCH_SIZE = 100
SAMPLES_PER_SONG = 61


def init():

    al_funcs = (al.input_greedy_sampling, al.output_greedy_sampling)

    # TODO: Regression trees
    ml_funcs = (ml.gradient_tree_boosting, ml.decision_tree)

    # Load Dataset 1
    storage.load_dataset(
        "ds1_train",
        Path("res/data/features_librosa_train_PCA.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    storage.load_dataset(
        "ds1_test",
        Path("res/data/features_librosa_test_PCA.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    # Load Dataset 2
    storage.load_dataset(
        "ds2_train",
        Path("res/data/features_librosa_train_VT.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    storage.load_dataset(
        "ds2_test",
        Path("res/data/features_librosa_test_VT.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    """
    # Load Dataset 3
    storage.load_dataset(
        "ds3_train",
        Path("res/data/features_librosa_train_D.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    storage.load_dataset(
        "ds3_test",
        Path("res/data/features_librosa_test_D.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )
    """

    datasets = ("ds1", "ds2")

    return [LearningProfile(f"{ds}_train", f"{ds}_test", al_func,
                            ml_func, BATCH_SIZE)
            for ds in datasets
            for al_func in al_funcs
            for ml_func in ml_funcs]


##########


learning_profiles = init()

viability_phase(learning_profiles)
