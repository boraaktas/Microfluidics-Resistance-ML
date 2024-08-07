import pandas as pd

from src.machine_learning import read_data, preprocess, train_test_split, train, test

if __name__ == '__main__':
    SIMULATION_RESULTS_CSVS_PATH: list[str] = ['drive_data/datasets/simulations/Simulation_Lib_5.csv',
                                               'drive_data/datasets/simulations/Simulation_Lib_6.csv']

    ALL_DATA: pd.DataFrame = read_data(SIMULATION_RESULTS_CSVS_PATH)
    LEN_DATA = len(ALL_DATA)
    print(f'Total number of drive_data points: {LEN_DATA}')

    # --------------------------------------- HARDCODES ---------------------------------------
    ALL_DATA['Simulation_Number'] = [i + 1 for i in range(LEN_DATA)]
    ALL_DATA = ALL_DATA.drop(columns=['Theoretical_Resistance',
                                      'Pressure_Difference',
                                      'Flow_Rate',
                                      'Step_Size',
                                      'Side_Length'])
    # -----------------------------------------------------------------------------------------

    FEATURE_COLUMN_NAMES = ALL_DATA.columns[2:-1]
    TARGET_COLUMN_NAME = ALL_DATA.columns[-1]

    print(f'Feature column names: {FEATURE_COLUMN_NAMES}')
    print(f'Target column name: {TARGET_COLUMN_NAME}')

    BASE_LEARNERS_PICKLE_PATH = 'drive_data/pickles/base_learner_pickles/'
    META_LEARNER_PICKLE_PATH = 'drive_data/pickles/meta_learner_pickles/'

    TRAINED_DATA_CSV_PATH = 'drive_data/datasets/prediction_outputs/'

    PREPROCESSED_DATA = preprocess(data=ALL_DATA,
                                   feature_column_names=FEATURE_COLUMN_NAMES)
    LEN_PREPROCESSED_DATA = len(PREPROCESSED_DATA)
    print(f'Number of drive_data points after preprocessing: {LEN_PREPROCESSED_DATA}')

    TRAIN_DATA, TEST_DATA = train_test_split(data=PREPROCESSED_DATA, frac_train=0.8)
    print(f'Training with {len(TRAIN_DATA)} drive_data points')
    print(f'Testing with {len(TEST_DATA)} drive_data points')

    TEST_DATA.to_csv(TRAINED_DATA_CSV_PATH + 'test_data.csv', index=False)

    TRAINED_DATA, BASE_MODELS_DICT, META_MODEL_TUPLE = train(data_df=TRAIN_DATA,
                                                             validation_percent=0.1,

                                                             feature_column_names=FEATURE_COLUMN_NAMES,
                                                             target_column_name=TARGET_COLUMN_NAME,

                                                             base_prediction_type='regression',
                                                             threshold_reg_adj_r2=0.9,
                                                             threshold_diff_percent=0.1,

                                                             save_output=True,
                                                             output_path=TRAINED_DATA_CSV_PATH,

                                                             save_models=True,
                                                             base_learners_pickle_path=BASE_LEARNERS_PICKLE_PATH,
                                                             meta_learner_pickle_path=META_LEARNER_PICKLE_PATH)

    BASE_MODELS_DICT_WITHOUT_RANK = {key: value[1] for key, value in BASE_MODELS_DICT.items()}

    TEST_RMSE, TEST_MAPE = test(data_df=TEST_DATA,
                                feature_column_names=FEATURE_COLUMN_NAMES,
                                target_column_name=TARGET_COLUMN_NAME,
                                base_models_dict=BASE_MODELS_DICT_WITHOUT_RANK,
                                meta_model_tuple=META_MODEL_TUPLE,
                                plot=True
                                )

    print(f'Test RMSE: {TEST_RMSE}')
    print(f'Test MAPE: {TEST_MAPE}')
