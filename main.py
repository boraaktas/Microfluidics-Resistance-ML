import pandas as pd
from src import read_data, preprocess, train_test_split, train, test

if __name__ == '__main__':
    SIMULATION_RESULTS_CSVS_PATH: list[str] = ['data/datasets/simulations/Simulation_Lib_1.csv',
                                               'data/datasets/simulations/Simulation_Lib_2.csv',
                                               'data/datasets/simulations/Simulation_Lib_3.csv',
                                               'data/datasets/simulations/Simulation_Lib_4.csv']

    ALL_DATA: pd.DataFrame = read_data(SIMULATION_RESULTS_CSVS_PATH)
    LEN_DATA = len(ALL_DATA)
    print(f'Total number of data points: {LEN_DATA}')

    # --------------------------------------- HARDCODES ---------------------------------------
    ALL_DATA['Simulation_Number'] = [i + 1 for i in range(LEN_DATA)]
    ALL_DATA = ALL_DATA.drop(columns=['Theoretical_Resistance',
                                      'Pressure_Difference',
                                      'Flow_Rate'])
    # -----------------------------------------------------------------------------------------

    FEATURE_COLUMN_NAMES = ALL_DATA.columns[2:-1]
    TARGET_COLUMN_NAME = ALL_DATA.columns[-1]

    BASE_LEARNERS_PICKLE_PATH = 'data/pickles/base_learner_pickles/'
    META_LEARNER_PICKLE_PATH = 'data/pickles/meta_learner_pickles/'

    TRAINED_DATA_CSV_PATH = 'data/datasets/prediction_outputs/'

    PREPROCESSED_DATA = preprocess(data=ALL_DATA,
                                   feature_column_names=FEATURE_COLUMN_NAMES)
    LEN_PREPROCESSED_DATA = len(PREPROCESSED_DATA)
    print(f'Number of data points after preprocessing: {LEN_PREPROCESSED_DATA}')

    TRAIN_DATA, TEST_DATA = train_test_split(data=PREPROCESSED_DATA, frac_train=0.8)
    print(f'Training with {len(TRAIN_DATA)} data points')
    print(f'Testing with {len(TEST_DATA)} data points')

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

    TEST_RMSE, TEST_MAPE = test(data_df=TEST_DATA,
                                feature_column_names=FEATURE_COLUMN_NAMES,
                                target_column_name=TARGET_COLUMN_NAME,
                                base_models_dict=BASE_MODELS_DICT,
                                meta_model_tuple=META_MODEL_TUPLE,
                                plot=True
                                )

    print(f'Test RMSE: {TEST_RMSE}')
    print(f'Test MAPE: {TEST_MAPE}')
