import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.error_metrics import RMSE, MAPE
from src.prediction_model import PredictionModel


def test(data_df: pd.DataFrame,
         feature_column_names: list[str],
         target_column_name: str,
         upload_model: bool = False,
         base_models_dict: dict[str, object] = None,
         meta_model_tuple: tuple = None,
         base_models_path: str = None,
         meta_model_path: str = None,
         plot: bool = False
         ) -> tuple[float, float]:
    """
    Test the trained models on the test data

    :param data_df: The test data
    :param feature_column_names: The names of the columns that are the features
    :param target_column_name: The name of the column that is the target
    :param upload_model: A boolean that indicates whether to upload the models or not

    :param base_models_dict: The best base models that were trained
    :param meta_model_tuple: The trained metamodel

    :param base_models_path: The path where the base models are saved
    :param meta_model_path: The path where the metamodel is saved

    :param plot: A boolean that indicates whether to plot the predictions or not

    :return: The RMSE and MAPE of the test data
    """
    if upload_model:
        try:
            predModel = PredictionModel(base_models_path, meta_model_path)
            base_models_dict = predModel.base_learners_dict
            meta_model_tuple = predModel.meta_learner
        except FileNotFoundError:
            raise FileNotFoundError('Please provide the correct paths for the models')

    if not upload_model:
        if base_models_dict is None or meta_model_tuple is None:
            raise ValueError('Please provide the best base models and the meta model, or set upload_model to True, '
                             'and provide the paths of the models')

    # Get the features and target
    X_TEST = data_df[feature_column_names]
    Y_TEST = data_df[target_column_name]

    predictions = []
    for index, row in X_TEST.iterrows():
        row = row.to_frame().T

        base_model_prediction = str(meta_model_tuple[1].predict(row)[0])
        # base_model_prediction = 'GaussianProcessRegressor'
        # base_model_prediction = 'RandomForestRegressor'
        # base_model_prediction = 'LGBMRegressor'
        # base_model_prediction = 'ExtraTreesRegressor'

        base_model = base_models_dict[base_model_prediction]
        prediction = float(base_model.predict(row)[0])

        print(f"{index} - "
              f"Base model prediction: {base_model_prediction}, "
              f"Prediction: {prediction}, "
              f"Real value: {Y_TEST[index]}")
        predictions.append(float(prediction))

    predictions = np.array(predictions)

    # Get the RMSE and MAPE of the test data
    rmse = RMSE(Y_TEST, predictions)
    mape = MAPE(Y_TEST, predictions)

    if plot:
        # draw a line as y=x
        x = np.linspace(min(Y_TEST), max(Y_TEST), 100)
        y = x
        plt.plot(x, y, color='red')
        plt.scatter(Y_TEST, predictions)
        plt.xlabel('Real Values')
        plt.ylabel('Predictions')

        plt.show()

    return rmse, mape
