from pathlib import Path
from learning_profile import LearningProfile
from api.storage import load_dataset
from api import al
from api import ml


BATCH_SIZE = 100


def init_phase():

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
    #   that this list can be iterated over when creating Learning Profiles.

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

    # Load Dataset 1
    load_dataset(
        "ds1_train",
        Path("res/data/features_librosa_train_PCA.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    load_dataset(
        "ds1_test",
        Path("res/data/features_librosa_test_PCA.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    # Load Dataset 2
    load_dataset(
        "ds2_train",
        Path("res/data/features_librosa_train_VT.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    load_dataset(
        "ds2_test",
        Path("res/data/features_librosa_test_VT.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    # Load Dataset 3
    load_dataset(
        "ds3_train",
        Path("res/data/features_librosa_train_D.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    load_dataset(
        "ds3_test",
        Path("res/data/features_librosa_test_D.npy"),
        Path("res/data/arousal_cont_average.csv"),
        Path("res/data/valence_cont_average.csv"),
        Path("res/data/arousal_cont_std.csv"),
        Path("res/data/valence_cont_std.csv")
    )

    # Change this to choose what datasets to evaluate.
    # datasets = ["ds1", "ds2", "ds3"]
    datasets = ["ds1", "ds2"]

    ###########################################################################
    # Initialize Learning Profiles
    ###########################################################################

    lps = [LearningProfile(f"{ds}_train", f"{ds}_test", al_func,
                           ml_func, hpars, BATCH_SIZE)
           for ds in datasets
           for al_func in al_funcs
           for (ml_func, hpars) in ml_funcs]

    return lps
