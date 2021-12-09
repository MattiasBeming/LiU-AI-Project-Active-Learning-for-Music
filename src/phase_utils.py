from enum import Enum


###############################################################################
# Evaluation Phase
###############################################################################

class Eval(Enum):
    """
    Enum for Evaluation mode.
    """
    AROUSAL = 0
    VALENCE = 1
    MEAN = 2


class lpParser:
    """
    Used when reading data from an .npy file to create a learning profile
    with getter functions for different parameters.

    Based on Evaluation/Presentation mode some paramters are set accordingly.
    """

    def __init__(self, id, profile, eval=None, pres=None):
        """
        Init function for lpParser.

        Args:
            id (String): id for the learning profile.
            profile (list): list of objects containing the data for the
                learning profile read from disk.
            eval (Enum): Enum for Evaluation mode. Defaults to None.
            pres (Enum): Enum for Presentation mode. Defaults to None.
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
            eval (Enum): Enum for evaluation mode.
        """
        self._eval = eval

        if eval == Eval.AROUSAL:
            self._score = self._score_arousal
            self._MSE = self._MSE_arousal
        elif eval == Eval.VALENCE:
            self._score = self._score_valence
            self._MSE = self._MSE_valence
        elif eval == Eval.MEAN:
            self._score = self._score_mean
            self._MSE = self._MSE_mean

    def set_pres_mode(self, pres):
        """
        Set the presentation mode.
        Updates the attribute parameter for the given presentation mode.

        Args:
            pres (Enum): Enum for presentation mode.
        """
        self._pres = pres

        if pres == Pres.AL:
            self._attr = self._al_func_name
        elif pres == Pres.ML:
            self._attr = self._ml_func_name
        elif pres == Pres.DS:
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


###############################################################################
# Presentation Phase
###############################################################################

class Pres(Enum):
    """
    Enum for Presentation mode.
    """
    AL = 0
    ML = 1
    DS = 2  # dataset
