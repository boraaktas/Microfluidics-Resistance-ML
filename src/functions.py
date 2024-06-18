import os
import pickle
import random
from typing import Union

import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier


def read_data(csv_files_paths: list[str]) -> pd.DataFrame:
    """
    Read data from csv files and concatenate them into a single dataframe
    Drop rows with NaN values and reset index
    :param csv_files_paths: list of csv files paths
    :return: concatenated dataframe
    """

    whole_data: pd.DataFrame = pd.DataFrame()

    for csv_file_path in csv_files_paths:
        current_data = pd.read_csv(csv_file_path)
        current_data = current_data.dropna().reset_index(drop=True)

        whole_data = pd.concat([whole_data, current_data], ignore_index=True)

    whole_data = whole_data.dropna().reset_index(drop=True)

    return whole_data


def train_test_split(data: pd.DataFrame,
                     frac_train: float) -> pd.DataFrame:
    data['train_or_test'] = None
    # put random values in the column select train with 0.9 probability

    for i, row in data.iterrows():
        data.loc[i, 'train_or_test'] = 'train' if random.random() < frac_train else 'test'

    # put train_or_test column at the beginning of the dataframe
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    return data


def lazy_fit(data: pd.DataFrame,
             feature_column_names: list[str],
             target_column_name: str,
             prediction_type: str,
             ) -> tuple[dict, pd.DataFrame]:
    x_train, y_train = (data[data['train_or_test'] == 'train'][feature_column_names],
                        data[data['train_or_test'] == 'train'][target_column_name])
    x_test, y_test = (data[data['train_or_test'] == 'test'][feature_column_names],
                      data[data['train_or_test'] == 'test'][target_column_name])

    if prediction_type == 'classification':
        lazy = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)

    elif prediction_type == 'regression':
        lazy = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)

    else:
        raise ValueError('prediction_type should be either "classification" or "regression"')

    lazy_scores, lazy_preds = lazy.fit(x_train, x_test, y_train, y_test)

    return lazy.models, lazy_scores


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean squared error (MSE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean squared error (MSE)
    """
    return float(np.mean((y_true - y_pred) ** 2))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the root mean squared error (RMSE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: root mean squared error (RMSE)
    """
    return float(np.sqrt(MSE(y_true, y_pred)))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean absolute error (MAE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean absolute error (MAE)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean absolute percentage error (MAPE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean absolute percentage error (MAPE)
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def get_comparison_df_for_base_models(base_models_dict: dict,
                                      base_scores_df: pd.DataFrame,
                                      train_data: pd.DataFrame,
                                      test_data: pd.DataFrame,
                                      feature_column_names: list[str],
                                      target_column_name: str
                                      ) -> pd.DataFrame:
    models_comparison_df = pd.DataFrame(columns=['Model', 'Adjusted R-Squared', 'R-Squared',
                                                 'Train_RMSE', 'Test_RMSE',
                                                 'Train_MAPE', 'Test_MAPE', 'Time Taken'])

    for model_name, model in base_models_dict.items():
        # predict the train data
        train_preds, train_mse, train_rmse, train_mae, train_mape = (
            predict_set_with_model(model,
                                   train_data[feature_column_names],
                                   train_data[target_column_name]))

        # predict the test data
        test_preds, test_mse, test_rmse, test_mae, test_mape = (
            predict_set_with_model(model,
                                   test_data[feature_column_names],
                                   test_data[target_column_name]))

        adjust_r2 = base_scores_df.loc[model_name]['Adjusted R-Squared']
        r2 = base_scores_df.loc[model_name]['R-Squared']
        time_taken = base_scores_df.loc[model_name]['Time Taken']

        new_row_dict = {'Model': model_name,
                        'Adjusted R-Squared': adjust_r2,
                        'R-Squared': r2,
                        'Train_RMSE': train_rmse,
                        'Test_RMSE': test_rmse,
                        'Train_MAPE': train_mape,
                        'Test_MAPE': test_mape,
                        'Time Taken': time_taken}

        models_comparison_df.loc[len(models_comparison_df)] = new_row_dict

    models_comparison_df = models_comparison_df.sort_values(by=['Adjusted R-Squared',
                                                                'Train_RMSE', 'Test_RMSE',
                                                                'Time Taken'],
                                                            ascending=False)
    models_comparison_df.reset_index(drop=True, inplace=True)

    return models_comparison_df


def predict_set_with_model(model,
                           data_X: pd.DataFrame,
                           data_Y: Union[pd.DataFrame, np.ndarray, list]
                           ) -> tuple[np.ndarray, float, float, float, float]:
    if isinstance(data_Y, pd.DataFrame):
        data_Y = data_Y.to_numpy()

    if isinstance(data_Y, list):
        data_Y = np.array(data_Y)

    # Make predictions
    preds: np.ndarray = model.predict(data_X)

    # Calculate the error
    mse = MSE(data_Y, preds)
    rmse = RMSE(data_Y, preds)
    mae = MAE(data_Y, preds)
    mape = MAPE(data_Y, preds)

    return preds, mse, rmse, mae, mape


def choose_best_base_models(threshold_adj_r2: float,
                            models_comp_df: pd.DataFrame,
                            models_dict: dict,
                            scores_df: pd.DataFrame
                            ) -> dict[str, tuple[int, object]]:
    # it is a dictionary that contains model name as key and tuple of model rank and model as value
    chosen_models: dict[str, tuple[int, object]] = {}

    if models_comp_df['Adjusted R-Squared'].max() < threshold_adj_r2:
        # chose best 5 models according to Adjusted R2
        msg = f'No model has Adjusted R-Squared larger than {threshold_adj_r2}. ' \
              f'Choosing the best 5 models according to Adjusted R-Squared.'
        print(msg)

        # update threshold
        threshold_adj_r2: float = models_comp_df['Adjusted R-Squared'].nlargest(5).min()

    # get the models that have larger or equal to the threshold Adjusted R2
    for model_name, model in models_dict.items():
        if scores_df.loc[model_name, 'Adjusted R-Squared'] >= threshold_adj_r2:
            current_model_rank: int = models_comp_df.Model.tolist().index(model_name)
            chosen_models[model_name] = (current_model_rank, model)

    # sort the chosen models according to their rank
    chosen_models = dict(sorted(chosen_models.items(), key=lambda item: item[1][0]))

    return chosen_models


def create_meta_learner_data(base_data: pd.DataFrame,
                             base_feature_column_names: list[str],
                             base_target_column_name: str,
                             best_base_models_dict: dict[str, tuple],
                             threshold_diff_percent: float
                             ) -> tuple[pd.DataFrame, list[str], str]:

    meta_data = base_data.copy()
    len_DATA = len(meta_data)

    # Calculate the error for each model for each data point
    for model_name in best_base_models_dict:
        model = best_base_models_dict[model_name][1]

        current_model_predictions = model.predict(meta_data[base_feature_column_names])

        # TODO: We need to choose which error metric to use
        diff = np.abs(current_model_predictions - meta_data[base_target_column_name])
        diff_percentage = (diff / meta_data[base_target_column_name]) * 100

        meta_data[model_name] = diff_percentage

    # Find the best model for each data point
    for i in range(len_DATA):

        error_values: dict[str, float] = {}
        for model_name in best_base_models_dict:
            error = meta_data.loc[i, model_name]
            error_values[model_name] = error

        # if there are models their difference is less than the threshold,
        # choose the one with the lowest rank in the CHOSEN_LAZY_REG_MODELS
        if any(error < threshold_diff_percent for error in error_values.values()):
            threshold_models_ranks: dict[str, int] = {}
            for threshold_model_name in error_values:
                if error_values[threshold_model_name] < threshold_diff_percent:
                    threshold_models_ranks[threshold_model_name] = best_base_models_dict[threshold_model_name][0]

            min_diff_model = min(threshold_models_ranks, key=threshold_models_ranks.get)

        else:
            min_diff_model = min(error_values, key=error_values.get)

        # add the best model for the data point
        meta_data.loc[i, 'Best_Model'] = min_diff_model

    # drop the columns that are not needed anymore
    meta_data = meta_data.drop(columns=[model_name for model_name in best_base_models_dict])

    # drop old target column
    meta_data = meta_data.drop(columns=[base_target_column_name])

    return meta_data, base_feature_column_names, 'Best_Model'


def choose_best_meta_model(meta_models_dict: dict,
                           meta_scores_df: pd.DataFrame
                           ) -> tuple[str, object]:
    # Choose the best model which has a higher accuracy
    best_model_name = meta_scores_df['Accuracy'].idxmax()
    best_model = meta_models_dict[best_model_name]

    model_tuple = (best_model_name, best_model)

    return model_tuple


def dump_all_chosen_models_to_pickle(base_learner_models_dict: dict,
                                     base_learner_features: list[str],
                                     base_learners_path: str,
                                     meta_learner_model_tuple: tuple,
                                     meta_learner_features: list[str],
                                     meta_learner_path: str) -> None:

    # before dumping, we need to remove all files and folders in the given paths
    if os.path.exists(base_learners_path):
        for file in os.listdir(base_learners_path):
            os.remove(f'{base_learners_path}{file}')

    if os.path.exists(meta_learner_path):
        for file in os.listdir(meta_learner_path):
            os.remove(f'{meta_learner_path}{file}')

    # dump chosen base learner models to pickle
    for model_name, model in base_learner_models_dict.items():
        pickle.dump(model[1], open(f'{base_learners_path}{model_name}.pkl', 'wb'))

    # dump chosen base learner features to a txt file
    with open(f'{base_learners_path}base_learner_features.txt', 'w') as f:
        for feature in base_learner_features:
            f.write(f'{feature}\n')

    # dump chosen meta learner model to pickle
    meta_learner_model_name = meta_learner_model_tuple[0]
    meta_learner_model = meta_learner_model_tuple[1]
    pickle.dump(meta_learner_model, open(f'{meta_learner_path}{meta_learner_model_name}.pkl', 'wb'))

    # dump chosen meta learner features to a txt file
    with open(f'{meta_learner_path}meta_learner_features.txt', 'w') as f:
        for feature in meta_learner_features:
            f.write(f'{feature}\n')


def save_outputs(output_path: str,
                 return_data_df: pd.DataFrame) -> None:
    """
    Save the data to a csv file
    :param output_path: path to save the output data
    :param return_data_df: data to be saved
    :return: None
    """
    return_data_df.to_csv(output_path, index=False)
