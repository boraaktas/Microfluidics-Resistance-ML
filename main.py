import pandas as pd
from src import read_data, train

if __name__ == '__main__':
    SIMULATION_RESULTS_CSVS_PATH: list[str] = ['data/datasets/simulations/Simulation_Lib_1.csv',
                                               'data/datasets/simulations/Simulation_Lib_2.csv',
                                               'data/datasets/simulations/Simulation_Lib_3.csv',
                                               'data/datasets/simulations/Simulation_Lib_4.csv']

    ALL_DATA: pd.DataFrame = read_data(SIMULATION_RESULTS_CSVS_PATH)
    LEN_DATA = len(ALL_DATA)
    ALL_DATA['Simulation_Number'] = [i + 1 for i in range(LEN_DATA)]

    # ------------------------------------ Preprocessing ------------------------------------
    ALL_DATA = ALL_DATA.drop(columns=['Theoretical_Resistance',
                                      'Pressure_Difference',
                                      'Flow_Rate'])

    print(f'Total number of data points: {LEN_DATA}')
    for row in range(LEN_DATA):
        if ALL_DATA.loc[row, 'Corner'] == 0:
            ALL_DATA.loc[row, 'Fillet_Radius'] = 0

        # Fillet_Radius, Width, and Height are in mm, convert them to micrometers
        ALL_DATA.loc[row, 'Fillet_Radius'] = ALL_DATA.loc[row, 'Fillet_Radius'] * 1000
        ALL_DATA.loc[row, 'Width'] = ALL_DATA.loc[row, 'Width'] * 1000
        ALL_DATA.loc[row, 'Height'] = ALL_DATA.loc[row, 'Height'] * 1000
    # -----------------------------------------------------------------------------------------

    TRAIN_DATA = ALL_DATA.copy()
    TRAIN_DATA = TRAIN_DATA.sample(n=25000, random_state=42).reset_index(drop=True)

    # select 5000 data points that is not chosen for the training data
    TEST_DATA = pd.DataFrame(columns=TRAIN_DATA.columns)
    for i in range(LEN_DATA):
        row = ALL_DATA.loc[i]
        sim_num = row['Simulation_Number']
        if sim_num not in TRAIN_DATA['Simulation_Number'].values:
            TEST_DATA.loc[len(TEST_DATA)] = row

    print(f'Total number of data points for real test: {len(TEST_DATA)}')
    TEST_DATA = TEST_DATA.sample(n=5000, random_state=42).reset_index(drop=True)

    TEST_DATA.to_csv('data/datasets/train_outputs/test_data.csv', index=False)
    # ------------------------------------ Training ------------------------------------
    FEATURE_COLUMN_NAMES = TRAIN_DATA.columns[2:-1]
    TARGET_COLUMN_NAME = TRAIN_DATA.columns[-1]

    BASE_LEARNERS_PICKLE_PATH = 'data/pickles/base_learner_pickles/'
    META_LEARNER_PICKLE_PATH = 'data/pickles/meta_learner_pickles/'

    TRAINED_DATA_CSV_PATH = 'data/datasets/train_outputs/train_data.csv'

    print(f'Training the models with {LEN_DATA} data points')
    TRAINED_DATA, BEST_BASE_MODELS_DICT, META_MODEL_TUPLE = train(data_df=TRAIN_DATA,
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
