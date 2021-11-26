from api import storage


class LearningProfile:

    def __init__(self, train_dataset_name: str, test_dataset_name: str,
                 al_func, ml_func):
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
        """
        self._train_dataset_name = train_dataset_name
        self._test_dataset_name = test_dataset_name
        self._al_func = al_func
        self._ml_func = ml_func

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
        return self._al_func

    def get_ml_function(self):
        return self._ml_func

    def __str__(self):
        return (
            f"LearningProfile("
            f"train='{self._train_dataset_name}', "
            f"test='{self._test_dataset_name}', "
            f"al_func={self._al_func.__name__}, "
            f"ml_func={self._ml_func.__name__})"
        )
