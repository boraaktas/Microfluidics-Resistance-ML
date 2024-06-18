import pandas as pd
from src import read_data, train

if __name__ == '__main__':
    SIMULATION_RESULTS_CSVS_PATH: list[str] = ['data/datasets/simulations/Simulation_Lib_1.csv',
                                               'data/datasets/simulations/Simulation_Lib_2.csv',
                                               'data/datasets/simulations/Simulation_Lib_3.csv',
                                               'data/datasets/simulations/Simulation_Lib_4.csv']

    WHOLE_DATA: pd.DataFrame = read_data(SIMULATION_RESULTS_CSVS_PATH)
    WHOLE_DATA = WHOLE_DATA.sample(n=10000, random_state=42).reset_index(drop=True)

    WHOLE_DATA = WHOLE_DATA.drop(columns=['Theoretical_Resistance',
                                          'Pressure_Difference',
                                          'Flow_Rate'])

    LEN_DATA = len(WHOLE_DATA)
    for row in range(LEN_DATA):
        if WHOLE_DATA.loc[row, 'Corner'] == 0:
            WHOLE_DATA.loc[row, 'Fillet_Radius'] = 0

        # Fillet_Radius, Width, and Height are in mm, convert them to micrometers
        WHOLE_DATA.loc[row, 'Fillet_Radius'] = WHOLE_DATA.loc[row, 'Fillet_Radius'] * 1000
        WHOLE_DATA.loc[row, 'Width'] = WHOLE_DATA.loc[row, 'Width'] * 1000
        WHOLE_DATA.loc[row, 'Height'] = WHOLE_DATA.loc[row, 'Height'] * 1000

    FEATURE_COLUMN_NAMES = WHOLE_DATA.columns[2:-1]
    TARGET_COLUMN_NAME = WHOLE_DATA.columns[-1]

    BASE_LEARNERS_PICKLE_PATH = 'data/pickles/base_learner_pickles/'
    META_LEARNER_PICKLE_PATH = 'data/pickles/meta_learner_pickles/'

    TRAINED_DATA_CSV_PATH = 'data/datasets/train_outputs/trained_data.csv'

    print(f'Training the models with {LEN_DATA} data points')
    TRAINED_DATA, BEST_BASE_MODELS_DICT, META_MODEL_TUPLE = train(data_df=WHOLE_DATA,
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
