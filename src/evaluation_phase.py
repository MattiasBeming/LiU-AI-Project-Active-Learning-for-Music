import numpy as np
from pathlib import Path
from os import listdir
from datetime import datetime
from phase_utils import Eval, lpParser

LP_FILE_NAME = 'lp'


###############################################################################
# Store data to disk - Phase
# Uses the class LearningProfile and not lpParser
###############################################################################

def create_datetime_subdir(dir_path):
    """
    Create subdirectory named after the current date and time.

    Args:
        dir_path (Path): Path to directory.

    Returns:
        dir_path (Path): New path to subdirectory.
    """
    if not dir_path.is_dir():
        # Create dir
        dir_path.mkdir()

    # Create subdirectory named after the current date and time
    date_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    dir_path = Path(dir_path / date_time)
    dir_path.mkdir()
    return dir_path


def evaluation_phase(dir_path, learning_profiles):
    """
    Calculates the score for each learning profile and saves the score
    together with the MSE in .npy format.

    Files are created with the name LP_FILE_NAME-id.npy.

    Args:
        dir_path (Path): Directory path for output from learning profiles.
        learning_profiles (list): List of learning profiles.
    """
    for lp in learning_profiles:
        data = create_data(lp)
        file_path = dir_path / f"{LP_FILE_NAME}-{lp.get_id()}.npy"
        np.save(Path(file_path), data)
    return


def create_data(lp):
    """
    Creates a data array for the learning profile.

    Data Format: train_dataset_name, test_dataset_name, al_func_name,
    ml_func_name, score_arousal, score_valence, mean_score,
    MSE_arousal, MSE_valence, MSE_mean, batch_size, hyper_parameters

    Args:
        lp (LearningProfile): Contains all the data needed.

    Returns:
        np.array: list of specified Data Format.
    """
    arousal = lp.get_MSE_arousal()
    valence = lp.get_MSE_valence()

    score_arousal = np.sum(arousal)
    score_valence = np.sum(valence)

    mean = [np.mean(i) for i in zip(*[arousal, valence])]
    mean_score = np.sum(mean)

    return np.array([lp.get_train_dataset_name(),
                     lp.get_test_dataset_name(),
                     lp.get_al_function().__name__,
                     lp.get_ml_function().__name__,
                     score_arousal, score_valence, mean_score,
                     arousal, valence, mean,
                     lp.get_batch_size(),
                     lp.get_hyper_parameters()],
                    dtype=object)


###############################################################################
# Load data from disk - Phase
# Uses the class lpParser and not LearningProfile
###############################################################################


def evaluate_all_profiles(learning_profiles=[],
                          eval=Eval.AROUSAL,
                          nr_models=-1):
    """
    Given a list of learning profiles, evaluate them and return the first
    'nr_models' profiles with the best score according to the eval method.

    Args:
        learning_profiles (list): List of learning profiles (using lpParser).
        eval (Enum): method of evaluation. Defaults to Eval.AROUSAL.
        nr_models (int): number of models to include in plot.
            Defaults to -1 (All models included).

    Returns:
        List(Tuple(String, float)): List of tuples containing the
        first nr_models with the best profiles in
        decending order (based on score) and their corresponding score.
    """
    # Set eval mode for all learning profiles
    [lp.set_eval_mode(eval) for lp in learning_profiles]

    # Sort according to best score
    sorted_list = sorted(learning_profiles, key=lambda lp: lp.get_score())

    best_LPs = [(f"Best profile: {lp.get_name(True, True)} " +
                 f" h_pars: {lp.get_hyper_parameters()}",
                 lp.get_score()) for lp in sorted_list]

    min_ = min(len(sorted_list), nr_models)
    return best_LPs[:min_] if nr_models > -1 else best_LPs


def load_all_learning_profiles(dir_path):
    """
    Load all learning profiles from disk and parse them using lpParser.

    Args:
        dir_path (Path): Path to directory containing learning profiles.

    Returns:
        learning_profiles (list): list of parsed learning profiles.
    """
    # Load all learning profiles in the given dir_path
    learning_profiles = []
    for file_name in listdir(dir_path):
        if LP_FILE_NAME in file_name:
            lp_loaded = np.load(Path(dir_path / file_name), allow_pickle=True)
            # Get id from the file name
            id = file_name.split(LP_FILE_NAME + '-', 1)[1]
            lp = lpParser(id, lp_loaded)
            learning_profiles.append(lp)
    return learning_profiles


def print_id_name_for_learning_profiles(learning_profiles):
    print("id -- Learning Profile")
    for lp in learning_profiles:
        print("id:", lp.get_id(), "--",
              f"{lp.get_name(True, False)}")
