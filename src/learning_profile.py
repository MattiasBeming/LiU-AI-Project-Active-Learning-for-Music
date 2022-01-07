from api import storage
import numpy as np
import hashlib


class LearningProfile:

    def __init__(self, train_dataset_name: str, test_dataset_name: str,
                 al_func, ml_func, hyper_parameters={}, batch_size: int = 5):
        """
        Creates a Learning Profile. A Learning Profile is used for bundling
        train/test datasets with an Active Learning function and a Machine
        Learning function.

        Args:
            train_dataset_name (str): The name of the dataset to use for
                training. Will be fetched from storage, and is therefore
                assumed to exist there.
            test_dataset_name (str): The name of the dataset to use for
                testing. Will be fetched from storage, and is therefore
                assumed to exist there.
            al_func (function): An Active Learning function to use, such as
                those found in `api/al.py`.
            ml_func (function): A Machine Learning function to use, such as
                those found in `api/ml.py`.
            hyper_parameters (dict): A dictionary of hyper parameters to use.
                Default is an empty dictionary, meaning standard
                hyper parameters will be used for the ml model.
            batch_size (int): The batch size to use when performing active
                learning. Default is 5.
        """
        self._train_dataset_name = train_dataset_name
        self._test_dataset_name = test_dataset_name
        self._al_func = al_func
        self._ml_func = ml_func
        self._batch_size = batch_size
        self._hyper_parameters = hyper_parameters
        self._MSE_arousal = None
        self._MSE_valence = None

    def get_id(self):
        """
        Returns a unique ID based on this learning profile's
        dataset names and its al/ml functions.

        Returns:
            str: The unique ID.
        """
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()

    def get_train_dataset(self):
        """
        Gets the train dataset associated with this Learning Profile's train
        dataset name from storage. This name can be fetched with
        get_train_dataset_name().

        Returns:
            api.storage.Dataset: This Learning Profile's train dataset.
        """
        return storage.get_dataset(self._train_dataset_name)

    def get_train_dataset_name(self):
        """
        Gets the train dataset name of this Learning Profile. Get the actual
        dataset with get_train_dataset().

        Returns:
            str: This Learning Profile's train dataset name.
        """
        return self._train_dataset_name

    def get_test_dataset(self):
        """
        Gets the test dataset associated with this Learning Profile's test
        dataset name from storage. This name can be fetched with
        get_test_dataset_name().

        Returns:
            str: This Learning Profile's test dataset.
        """
        return storage.get_dataset(self._test_dataset_name)

    def get_test_dataset_name(self):
        """
        Gets the test dataset name of this Learning Profile. Get the actual
        dataset with get_test_dataset().

        Returns:
            str: This Learning Profile's test dataset name.
        """
        return self._test_dataset_name

    def get_al_function(self):
        """
        Gets the Active Learning function of this learning profile.

        Returns:
            function: The Active Learning function.
        """
        return self._al_func

    def get_ml_function(self):
        """
        Gets the Machine Learning function of this learning profile.

        Returns:
            function: The Machine Learning function.
        """
        return self._ml_func

    def get_batch_size(self):
        """
        Gets the batch size of this learning profile.

        Returns:
            int: The batch size.
        """
        return self._batch_size

    def set_hyper_parameters(self, **hyper_parameters):
        """
        All keyword arguments passed to this method are saved as hyper
        parameters. These values should be used in the Machine Learning
        function of this learning profile. See `get_ml_function()`.
        """
        self._hyper_parameters = hyper_parameters

    def get_hyper_parameters(self):
        """
        Returns the hyper parameters of this learning profile, that are to be
        used for its Machine Learning function. See `get_ml_function()`.

        Returns:
            dict: The hyper parameters on the form of a dictionary, with keys
                as parameter names mapped to their values.
        """
        return self._hyper_parameters

    def set_MSE(self, MSE_arousal: np.ndarray, MSE_valence: np.ndarray):
        """
        Sets the MSE values for this Learning Profile.

        Args:
            MSE_arousal (np.ndarray): MSE values for arousal.
            MSE_valence (np.ndarray): MSE values for valence.
        """
        self._MSE_arousal = MSE_arousal
        self._MSE_valence = MSE_valence

    def get_MSE_arousal(self):
        """
        Gets the MSE values for arousal.

        Returns:
            np.ndarray: MSE values for arousal, or `None` if the values
                have not yet been set through `set_MSE()`.
        """
        return self._MSE_arousal

    def get_MSE_valence(self):
        """
        Gets the MSE values for valence.

        Returns:
            np.ndarray: MSE values for valence, or `None` if the values
                have not yet been set through `set_MSE()`.
        """
        return self._MSE_valence

    def get_name(self):
        return (
            f"{self._train_dataset_name}, "
            f"{self._test_dataset_name}, "
            f"{self._al_func.__name__}, "
            f"{self._ml_func.__name__}, "
            f"{self._hyper_parameters}"
        )

    def __str__(self):
        return (
            f"LearningProfile("
            f"train='{self._train_dataset_name}', "
            f"test='{self._test_dataset_name}', "
            f"al_func={self._al_func.__name__}, "
            f"ml_func={self._ml_func.__name__}), "
            f"h_pars={self._hyper_parameters})"
        )
