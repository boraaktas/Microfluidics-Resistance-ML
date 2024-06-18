import os
import pickle

import pandas as pd


class PredictionModel:

    def __init__(self,
                 base_learners_pickle_path: str,
                 meta_learner_pickle_path: str):

        self.base_learners_dict: dict = PredictionModel.load_base_learners(base_learners_pickle_path)
        self.base_learner_features: list[str] = PredictionModel.load_base_learner_features(base_learners_pickle_path)

        self.meta_learner = PredictionModel.load_meta_learner(meta_learner_pickle_path)
        self.meta_learner_features: list[str] = PredictionModel.load_meta_learner_features(meta_learner_pickle_path)

    def predict(self,
                data_point_dict: dict[str, float]) -> float:
        """
        Predicts the target value of the given data point.

        :param data_point_dict: A dictionary that contains the data point. It should have the same keys as the feature
                                                                           column names.
        :return: prediction: The prediction of the data point.
        """

        # if keys of the data_point_dict does not include all the feature column names, raise an error

        # make dict to pandas dataframe to predict in the meta learner
        meta_data_point_df = pd.DataFrame(columns=self.meta_learner_features)
        for column_name in self.meta_learner_features:
            meta_data_point_df[column_name] = [data_point_dict[column_name]]

        # first predict the base learner by using the meta learner
        chosen_base_learner_name = str(self.meta_learner.predict(meta_data_point_df)[0])

        # get the predicted model from the base learners dictionary
        chosen_base_learner = self.base_learners_dict[chosen_base_learner_name]

        # make dict to pandas dataframe to predict in the base learner
        base_data_point_df = pd.DataFrame(columns=self.base_learner_features)
        for column_name in self.base_learner_features:
            base_data_point_df[column_name] = [data_point_dict[column_name]]

        prediction = float(chosen_base_learner.predict(base_data_point_df)[0])

        return prediction, chosen_base_learner_name

    @staticmethod
    def load_base_learners(base_learners_pickle_path: str) -> dict:
        """
        Loads the base learners from the pickle files in the given path.

        :param base_learners_pickle_path: The path where the base learners are saved.
        :return: base_learners_dict: A dictionary that contains the base learners. The key is the model name and,
                                     the value is the model itself.
        """

        base_learners_dict: dict[str, object] = {}
        for file in os.listdir(base_learners_pickle_path):
            if file.endswith('.pkl'):
                current_model_name: str = file.split('.')[0]
                current_model = pickle.load(open(base_learners_pickle_path + file, 'rb'))

                base_learners_dict[current_model_name] = current_model

        return base_learners_dict

    @staticmethod
    def load_meta_learner(meta_learner_pickle_path: str):
        """
        Loads the meta learner from the pickle file in the given path.

        :param meta_learner_pickle_path: The path where the meta learner is saved.
        :return: meta_learner: The meta learner model.
        """

        # if there is more than one pickle file and one txt file in the given path, raise an error
        if len(os.listdir(meta_learner_pickle_path)) > 2:
            raise ValueError('There is more than one pickle file in the given path. '
                             'Please provide the path to the exact pickle file.')

        pickle_file_path = None
        for file in os.listdir(meta_learner_pickle_path):
            if file.endswith('.pkl'):
                pickle_file_path = file

        meta_learner = pickle.load(open(meta_learner_pickle_path + pickle_file_path, 'rb'))

        return meta_learner

    @staticmethod
    def load_base_learner_features(base_learners_pickle_path: str) -> list[str]:

        base_learner_features = []
        # from the txt file in the path, get the feature names
        with open(base_learners_pickle_path + 'base_learner_features.txt', 'r') as f:
            for line in f:
                base_learner_features.append(line.strip())

        return base_learner_features

    @staticmethod
    def load_meta_learner_features(meta_learner_pickle_path: str) -> list[str]:

        meta_learner_features = []
        # from the txt file in the path, get the feature names
        with open(meta_learner_pickle_path + 'meta_learner_features.txt', 'r') as f:
            for line in f:
                meta_learner_features.append(line.strip())

        return meta_learner_features
