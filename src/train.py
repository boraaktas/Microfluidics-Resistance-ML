import os

import pandas as pd

from src.train_functions import (train_valid_split,
                                 lazy_fit,
                                 get_comparison_df_for_base_models,
                                 choose_best_base_models,
                                 create_meta_learner_data,
                                 choose_best_meta_model,
                                 dump_all_chosen_models_to_pickle,
                                 save_outputs)


def train(data_df: pd.DataFrame,
          validation_percent: float,

          feature_column_names: list[str],
          target_column_name: str,

          base_prediction_type: str = 'regression',
          threshold_reg_adj_r2: float = 0.9,
          threshold_diff_percent: float = 0.1,

          save_output: bool = False,
          output_path: str = None,

          save_models: bool = False,
          base_learners_pickle_path: str = None,
          meta_learner_pickle_path: str = None
          ) -> tuple[pd.DataFrame, dict[str, tuple[int, object]], tuple[str, object]]:
    """
    This function is used to train the models and save them to pickle files.

    :param data_df: The data that will be used to train the models.
    :param validation_percent: The percentage of the data that will be used for validation.

    :param feature_column_names: The names of the columns that will be used as features.
    :param target_column_name: The name of the column that will be used as target.

    :param base_prediction_type: The type of prediction that the base learners will be used for.

    :param threshold_reg_adj_r2: The threshold for adjusted R2 score to choose the best models for base learners.
    :param threshold_diff_percent: The threshold for the difference percentage between the true values and the predicted
                                   values for every data point to choose the best models for each data point.

    :param save_output: A boolean that indicates whether to save the outputs or not.
    :param output_path: The path where the outputs will be saved.

    :param save_models: A boolean that indicates whether to save the models or not.
    :param base_learners_pickle_path: The path where the base learners will be saved.
    :param meta_learner_pickle_path: The path where the meta learner will be saved.

    :return: return_data_df: The data that will be returned. It is the same as the input data with the addition of the
    :return: best_base_models_dict: A dictionary that contains the best base learners. The key is the model name and,
                                    the value is a tuple that contains the rank of the model and the model itself.
    :return: meta_model_tuple: A tuple that contains the best meta learner. The first element is the model name and,
                               the second element is the model itself.
    """

    if validation_percent < 0 or validation_percent > 1:
        msg = 'The validation_percentage must be between 0 and 1.'
        raise ValueError(msg)

    if save_models:
        if base_learners_pickle_path is None or meta_learner_pickle_path is None:
            msg = 'You must provide the paths for the base learners and the meta learner to save the models.'
            raise ValueError(msg)

        # if the path does not exist, raise an error
        if not os.path.exists(base_learners_pickle_path):
            raise ValueError(f'The path {base_learners_pickle_path} does not exist.')

        # if the path does not exist, raise an error
        if not os.path.exists(meta_learner_pickle_path):
            raise ValueError(f'The path {meta_learner_pickle_path} does not exist.')

    if save_output:
        if output_path is None:
            msg = 'You must provide the path to save the outputs.'
            raise ValueError(msg)

        # if the path does not exist, raise an error
        if not os.path.exists(output_path):
            raise ValueError(f'The path {output_path} does not exist.')

    if base_prediction_type not in ['regression', 'classification']:
        msg = 'The base_prediction_type must be either regression or classification.'
        raise ValueError(msg)

    base_data: pd.DataFrame = data_df

    base_feature_column_names = feature_column_names.copy()
    base_target_column_name = target_column_name

    # ---------------------------------------- TRAIN VALIDATION SPLIT --------------------------------------------------
    base_data = base_data.sample(frac=1).reset_index(drop=True)
    base_data = train_valid_split(base_data, frac_train=1 - validation_percent)
    base_train_data, base_valid_data = (base_data[base_data['train_or_valid'] == 'train'],
                                        base_data[base_data['train_or_valid'] == 'valid'])

    # --------------------------------------------- LAZY REGRESSION MODELS ---------------------------------------------
    base_models_dict, base_scores_df = lazy_fit(data=base_data,
                                                feature_column_names=base_feature_column_names,
                                                target_column_name=base_target_column_name,
                                                prediction_type=base_prediction_type)

    # Get comparison dataframe for base models
    base_models_comp_df = get_comparison_df_for_base_models(base_models_dict=base_models_dict,
                                                            base_scores_df=base_scores_df,
                                                            train_data=base_train_data,
                                                            valid_data=base_valid_data,
                                                            feature_column_names=base_feature_column_names,
                                                            target_column_name=base_target_column_name)

    # --------------------------------------------- CHOSEN_LAZY_REG_MODELS ---------------------------------------------
    # it is a dictionary that contains model name as key and tuple of model rank and model as value
    best_base_models_dict: dict[str, tuple[int, object]] = choose_best_base_models(
        threshold_adj_r2=threshold_reg_adj_r2,
        models_comp_df=base_models_comp_df,
        models_dict=base_models_dict,
        scores_df=base_scores_df)

    # ---------------------------------------- ADD BEST MODEL FOR EACH DATAPOINT ---------------------------------------
    meta_data, meta_feature_column_name, meta_target_column_name = (
        create_meta_learner_data(base_data=base_data,
                                 base_feature_column_names=base_feature_column_names,
                                 base_target_column_name=base_target_column_name,
                                 best_base_models_dict=best_base_models_dict,
                                 threshold_diff_percent=threshold_diff_percent))

    # --------------------------------------------- META LEARNER MODELS ------------------------------------------------
    meta_models_dict, meta_scores_df = lazy_fit(data=meta_data,
                                                feature_column_names=meta_feature_column_name,
                                                target_column_name=meta_target_column_name,
                                                prediction_type='classification')

    # choose the best model to use as meta learner
    meta_model_tuple = choose_best_meta_model(meta_models_dict=meta_models_dict,
                                              meta_scores_df=meta_scores_df)

    # --------------------------------------------- RETURN DATA --------------------------------------------------------
    return_data_df = base_data.copy()
    # put the meta_target_column_name to the return_data_df
    return_data_df[meta_target_column_name] = meta_data[meta_target_column_name]

    # ------------------------------------------------ SAVING ----------------------------------------------------------
    if save_models:
        dump_all_chosen_models_to_pickle(base_learner_models_dict=best_base_models_dict,
                                         base_learner_features=base_feature_column_names,
                                         base_learners_path=base_learners_pickle_path,
                                         meta_learner_model_tuple=meta_model_tuple,
                                         meta_learner_features=meta_feature_column_name,
                                         meta_learner_path=meta_learner_pickle_path)

    if save_output:
        save_outputs(output_path=output_path,
                     return_data_df=return_data_df)

    return return_data_df, best_base_models_dict, meta_model_tuple
