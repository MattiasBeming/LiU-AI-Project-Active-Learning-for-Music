from enum import Enum
from pathlib import Path
import json
import numpy as np

###############################################################################
# Classes
###############################################################################


class EvaluationMode(Enum):
    """
    Enum for Evaluation mode.
    """
    AROUSAL = 0
    VALENCE = 1
    MEAN = 2


class PresentationMode(Enum):
    """
    Enum for Presentation mode.
    """
    AL = 0
    ML = 1
    DS = 2  # dataset


class LearningProfileDescription:
    """
    Used when reading data from an .npy file to create a learning profile
    with getter functions for different parameters.

    Based on Evaluation/Presentation mode some paramters are set accordingly.
    """

    def __init__(self, id, profile, eval=None, pres=None):
        """
        Init function for LearningProfileDescription.

        Args:
            id (String): id for the learning profile.
            profile (list): list of objects containing the data for the
                learning profile read from disk.
            eval (EvaluationMode): Enum for Evaluation mode. Defaults to None.
            pres (PresentationMode): Enum for Presentation mode.
                Defaults to None.
        """
        self._id = id
        self._batch_size = profile[10]
        self._hyper_parameters = profile[11]

        # If needed:
        self._score_arousal = profile[4]
        self._score_valence = profile[5]
        self._score_mean = profile[6]

        self._MSE_arousal = profile[7]
        self._MSE_valence = profile[8]
        self._MSE_mean = profile[9]

        self._train_dataset_name = profile[0]
        self._test_dataset_name = profile[1]
        self._al_func_name = profile[2]
        self._ml_func_name = profile[3]

        # Evaluation mode
        self._eval = eval
        if self._eval is not None:
            self.set_eval_mode(self._eval)
        else:
            self._score = None
            self._MSE = None

        # Presentation mode
        self._pres = pres
        if self._pres is not None:
            self.set_pres_mode(self._pres)
        else:
            self._attr = None

    def set_eval_mode(self, eval):
        """
        Set the evaluation mode.
        Updates the score and MSE parameters for the given evalution.

        Args:
            eval (EvaluationMode): Enum for evaluation mode.
        """
        self._eval = eval

        if eval == EvaluationMode.AROUSAL:
            self._score = self._score_arousal
            self._MSE = self._MSE_arousal
        elif eval == EvaluationMode.VALENCE:
            self._score = self._score_valence
            self._MSE = self._MSE_valence
        elif eval == EvaluationMode.MEAN:
            self._score = self._score_mean
            self._MSE = self._MSE_mean

    def set_pres_mode(self, pres):
        """
        Set the presentation mode.
        Updates the attribute parameter for the given presentation mode.

        Args:
            pres (PresentationMode): Enum for presentation mode.
        """
        self._pres = pres

        if pres == PresentationMode.AL:
            self._attr = self._al_func_name
        elif pres == PresentationMode.ML:
            self._attr = self._ml_func_name
        elif pres == PresentationMode.DS:
            self._attr = self._train_dataset_name

    def get_id(self):
        return self._id

    def get_batch_size(self):
        return self._batch_size

    def get_score_arousal(self):
        return self._score_arousal

    def get_score_valence(self):
        return self._score_valence

    def get_score_mean(self):
        return self._score_mean

    def get_MSE_arousal(self):
        return self._MSE_arousal

    def get_MSE_valence(self):
        return self._MSE_valence

    def get_MSE_mean(self):
        return self._MSE_mean

    def get_train_dataset_name(self):
        return self._train_dataset_name

    def get_test_dataset_name(self):
        return self._test_dataset_name

    def get_al_func_name(self):
        return self._al_func_name

    def get_ml_func_name(self):
        return self._ml_func_name

    def get_name(self, align=False, include_batch_size=False):
        """
        Returns the concatinated string of the learning profile, containing
        training_dataset_name, test_dataset_name, al_func_name, ml_func_name
        and an optional batch_size.

        Decent alignment can be achieved by setting align to True.

        Args:
            align (bool): align text. Defaults to False.
            include_batch_size (bool): include batch size information.
                Defaults to False.

        Returns:
            String: A concatinated string of the learning profile.
        """
        if align:
            return f"{self._train_dataset_name : <10} " + \
                f"{self._test_dataset_name : <10} " + \
                f"{self._al_func_name : <35} " + \
                f"{self._ml_func_name : <30}" + \
                ("" if not include_batch_size else
                 f"{f'(Batch Size: {self._batch_size})' : >10}")

        return self._train_dataset_name + " " + \
            self._test_dataset_name + " " + \
            self._al_func_name + " " + \
            self._ml_func_name + " " + \
            ("" if not include_batch_size else
             f"(Batch Size: {self._batch_size})")

    def get_score(self):
        if self._eval is not None:
            return self._score
        else:
            raise ValueError("Evaluation mode not set.")

    def get_MSE(self):
        if self._eval is not None:
            return self._MSE
        else:
            raise ValueError("Evaluation mode not set.")

    def get_attr(self):
        if self._pres is not None:
            return self._attr
        else:
            raise ValueError("Presentation mode not set.")

    def get_hyper_parameters(self):
        return self._hyper_parameters

    def __str__(self):
        return f"lp-{self._id}"


class AnnotationStation:
    """
    Used for saving label annotations for songs.
    Reducing the need for annotating the same song more than once.

    Annotations will be saved as a dictionsary in following format
    for song ids 1 and 2::

        {'1': [[arousal1], [valence1]], '2': [[arousal2], [valence2]]}
    """

    def __init__(self, path: Path):
        """
        AnnotationStation constructor.

        Args:
            path (Path): Path to dictionary in json format.
                    (Note: include `.json` tag.)

        Raises:
            FileNotFoundError: When path is not correct.
        """
        self.path = path
        if path.exists():
            if path.is_file():
                with open(path, "r") as f:
                    self.annotations = json.loads(f.read())
            else:
                raise FileNotFoundError(f"{path} not a file!")
        else:
            self.annotations = dict()

    def is_song_id_in_annotations(self, song_id: int):
        """
        Checks if `song_id` is already saved in annotations and
        thus already annotated.

        Args:
            song_id (int): The song id to check for.

        Returns:
            bool: True if song id is in annotations.
        """
        return str(song_id) in self.annotations

    def add_annotation(self, song_id: int, arousal: np.ndarray,
                       valence: np.ndarray):
        """
        Add annotation of `song_id` to annotations dictionary,
        followed by saving it to file using `save_annotations()`.

        Args:
            song_id (int): The song id to add.
            arousal (np.ndarray): Dynamic arousal annotations, as a
                column vector.
            valence (np.ndarray): Dynamic valence annotations, as a
                column vector.
        """
        self.annotations[str(song_id)] = np.array([arousal, valence]).tolist()
        self.save_annotations()

    def get_annotations(self):
        return self.annotations

    def save_annotations(self):
        """
        Save annotations to the json file specified
        during construction.
        """
        with open(self.path, 'w') as f:
            json.dump(self.annotations, f)

###############################################################################
# Functions
###############################################################################


def get_specific_learning_profiles(learning_profiles=[],
                                   pres=PresentationMode.AL):
    """
    Yield a list of learning profiles for every attribute for the given
    Presentation mode.

    Args:
        learning_profiles (list): List of learning profiles
            (using LearningProfileDescription).
        pres (PresentationMode): presentation mode.
            Defaults to PresentationMode.AL.

    Yields:
        list(LearningProfileDescription): The list of learning profiles.
    """
    if not learning_profiles:
        raise ValueError("No learning profiles given")

    # Set pres mode for all learning profiles
    [lp.set_pres_mode(pres) for lp in learning_profiles]

    # Loop through all attributes and yield the models with the same attribute
    for attr in set([lp.get_attr() for lp in learning_profiles]):
        # Retrieve all learning profiles with the same attribute
        yield [lp for lp in learning_profiles if lp.get_attr() == attr]


def sort_by_score(learning_profiles=[],
                  eval=EvaluationMode.MEAN,
                  nr_models=-1):
    """
    Given a list of learning profiles, sort them by score according to
    the eval method and return the first 'nr_models' profiles.

    Args:
        learning_profiles (list): List of learning profiles
            (using LearningProfileDescription).
        eval (EvaluationMode): method of evaluation.
            Defaults to EvaluationMode.MEAN.
        nr_models (int): number of models to include in plot.
            Defaults to -1 (All models included).

    Returns:
        list: List of learning profiles containing the first 'nr_models'
            with the best profiles in decending order (based on score).
    """
    if not learning_profiles:
        raise ValueError("No learning profiles given")

    # Set eval mode for all learning profiles
    [lp.set_eval_mode(eval) for lp in learning_profiles]

    # Sort according to best score
    sorted_list = sorted(learning_profiles, key=lambda lp: lp.get_score())

    min_ = min(len(sorted_list), nr_models)
    return sorted_list[:min_] if nr_models > -1 else sorted_list


def retrieve_best_learning_profiles(learning_profiles=[], nr_models=-1):
    """
    Retrieve the 'nr_models' best learning profiles for all presentation modes.
    Results are presented in a dictionary on the following format::

        {
            AL: {
                'input_greedy_sampling': [lp1, lp2, lp3],
                'output_greedy_sampling': [lp10, lp11, lp12]
            },
            ML: {
                'linear_regression': [lp2, lp5, lp8],
                'decision_tree': [lp20, lp22, lp24]
            }, ...
        }
    where the list for a specific method is sorted by score and contains the
    'nr_models' best learning profiles.

    Args:
        learning_profiles (list): List of learning profiles
            (using LearningProfileDescription).
        nr_models (int): number of models to include in plot.
            Defaults to -1 (All models included).

    Returns:
        dict: Dictionary containing the best learning profiles for each
            presentation mode.
    """
    best_profiles = {}

    # Get the specific learning profiles for all presentation modes
    for pres_mode in PresentationMode:
        pres_name = pres_mode.name
        pres_mode_LPs = get_specific_learning_profiles(
            learning_profiles, pres_mode)

        pres_dict = {}
        for lps in pres_mode_LPs:
            # Get name of attribute (get_specific_learning_profiles() sets
            # pres_mode to enable the use of get_attr())
            attr = lps[0].get_attr()

            # Retrieve the 'nr_models' best learning profiles for
            # evaluation mode MEAN.
            pres_dict[attr] = sort_by_score(
                lps, EvaluationMode.MEAN, nr_models)

        best_profiles[pres_name] = pres_dict
    return best_profiles


def print_id_name_for_learning_profiles(learning_profiles):
    print("id -- Learning Profile")
    for lp in learning_profiles:
        print("id:", lp.get_id(), "--", f"{lp.get_name(True, False)}")
