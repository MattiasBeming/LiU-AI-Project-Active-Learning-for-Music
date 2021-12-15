from pathlib import Path
from learning_profile import LearningProfile
from api.transformer import format_pred_to_prior
from viability_phase import viability_phase
from api.storage import load_dataset, get_dataset
from api import al
from api import ml
import numpy as np


BATCH_SIZE = 100


def init_phase(sliding_window_length, model_eval=False):
    """
    Initializes the Learning Profiles to be evaluated.

    Args:
        sliding_window_length (int): The length of the sliding-window.
        model_eval (bool, optional): A flag that signals whether to do
            model selection or model evaluation. Defaults to False.

    Returns:
        list: Contains all the initialized Learning Profiles.
    """

    ###########################################################################
    # Initialize Al methods
    ###########################################################################

    al_funcs = [al.input_greedy_sampling, al.output_greedy_sampling,
                al.input_output_greedy_sampling]

    ###########################################################################
    # Initialize ML functions with hyperparameters
    ###########################################################################

    # ml_funcs consists of a list of tuples (ml_function, hyper_parameters).
    # The idea is to add tuples with various functions and hyperparameters such
    #    that this list can be iterated over when creating Learning Profiles.

    # K_neighbors
    kn_hpars = [3, 5, 7]
    ml_funcs = [(ml.k_neighbors, {"n_neighbors": hpar}) for hpar in kn_hpars]

    # Linear regression
    ml_funcs += [(ml.linear_regression, {})]

    # Gradient tree boosing
    grad_tree_hpars = [[learning_rate, n_estimators]
                       for learning_rate in [0.05, 0.1, 0.2]
                       for n_estimators in [50, 100, 200]]
    ml_funcs += [(ml.gradient_tree_boosting,
                 {"learning_rate": hpar[0], "n_estimators":hpar[1]})
                 for hpar in grad_tree_hpars]

    # Decision tree
    dec_tree_hpars = [None]
    ml_funcs += [(ml.decision_tree, {"max_depth": hpar})
                 for hpar in dec_tree_hpars]

    # Neural network
    nn_hpars = [20, 50]
    ml_funcs += [(ml.neural_network,
                  {"hidden_layer_sizes": hpar, "alpha": 1, "max_iter": 1000})
                 for hpar in nn_hpars]

    ###########################################################################
    # Load datasets
    ###########################################################################

    # Model selection
    train = "train"
    test = "val"

    # Model evaluation
    if model_eval:
        train = "trainval"
        test = "test"

    for i in range(1, 8, 2):
        load_train_test_datasets(i, train, test,  "PCA")
        load_train_test_datasets(i+1, train, test,  "VT")

    # ds1-2:
    # ds1 - PCA: train: no sw | test: no sw
    # ds2 - VT: train: no sw | test: no sw

    # ds3 - PCA: sw w/ prior = 0 | test: prior = 0
    apply_sliding_window(3, sliding_window_length, np.array([0]))

    # ds4 - VT: sw w/ prior = 0 | test: prior = 0
    apply_sliding_window(4, sliding_window_length, np.array([0]))

    # ds5 - PCA: sw w/ prior = T | test: prior = 0
    apply_sliding_window(5, sliding_window_length, np.array([]))

    # ds6 - VT: sw w/ prior = T | test: prior = 0
    apply_sliding_window(6, sliding_window_length, np.array([]))

    # ds7 - PCA: sw w/ prior = T | test: prior from basic regressor
    apply_sliding_window(7, sliding_window_length, np.array([]), True)

    # ds8 - VT: sw w/ prior = T | test: prior from basic regressor
    apply_sliding_window(8, sliding_window_length, np.array([]), True)

    # Change this to choose what datasets to evaluate.
    datasets = ["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"]

    ###########################################################################
    # Initialize Learning Profiles
    ###########################################################################

    lps = [LearningProfile(f"{ds}_train", f"{ds}_test", al_func,
                           ml_func, hpars, BATCH_SIZE)
           for ds in datasets
           for al_func in al_funcs
           for (ml_func, hpars) in ml_funcs]

    return lps


def load_train_test_datasets(id, train, test, feat_sel):
    """
    Loads train and test dataset into name "ds{id}_train" and "ds{id}_test".

    Args:
        id (int): Desired id for the dataset.
        train (str): String identifying what data we use as training data
            (either "train" or "trainval).
        test (str): String identifying what data we use as testing data.
            (either "val" or "test).
        feat_sel (str): String identifying what method of feature selection to
            pick (either "PCA" or "VT").
    """
    load_dataset(
        f"ds{id}_train",
        Path(
            f"res/data/features_librosa_{train}_{feat_sel}.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )
    load_dataset(
        f"ds{id}_test",
        Path(f"res/data/features_librosa_{test}_{feat_sel}.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )


def apply_sliding_window(id, window_len, train_prior, use_reg_prior=False):
    """
    Applies sliding window to train and test dataset with name="ds{id}".

    Args:
        id (int): Id of the dataset to apply sliding window to.
        window_len (int): Length of sliding window.
        train_prior (np.ndarray): Prior to use for first samples in
            'sliding_window_train()',
        use_reg_prior (bool, optional): A flag determining if
            linear regression will be used as prior for the test dataset.
            Defaults to False.
    """
    ds_train = get_dataset(f"ds{id}_train")
    temp_regressor = ml.linear_regression(ds_train)
    ds_train.sliding_window_train(window_len, train_prior)
    ds_test = get_dataset(f"ds{id}_test")
    if use_reg_prior:
        prior_feat = ds_test.get_first_n_samples_of_each_song(window_len)
        print(prior_feat.shape)
        test_prior = format_pred_to_prior(
            temp_regressor.predict(prior_feat), window_len)
    else:
        test_prior = np.array([0])
    ds_test.sliding_window_test(
        ml.linear_regression(ds_train), 5, test_prior)
