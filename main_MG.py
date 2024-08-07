from src.machine_learning import PredictionModel
from src.generative_model import GenerativeModel
from src.maze_functions import (random_maze_generator,
                                extract_features,
                                plot_maze)
from src.modelling_3D import build_3d_maze


def random_maze():
    STEP_SIZE_FACTOR = 0.5
    SIDE_LENGTH = 20
    TARGET_LOC_MODE = "east"  # east or north
    WIDTH, HEIGHT, FILLET_RADIUS = 0.10, 0.10, 0.10
    PATH_FINDING_MODE = "random"

    MAZE = random_maze_generator(SIDE_LENGTH, TARGET_LOC_MODE, PATH_FINDING_MODE)
    plot_maze(MAZE, print_maze=True)

    FEATURES_DICT = extract_features(MAZE, STEP_SIZE_FACTOR, WIDTH, HEIGHT, FILLET_RADIUS)
    print(FEATURES_DICT)

    maze = build_3d_maze(maze=MAZE, step_size_factor=STEP_SIZE_FACTOR,
                         width=WIDTH, height=HEIGHT, fillet_radius=FILLET_RADIUS)
    # maze.show()

    PREDICTION_MODEL = PredictionModel(base_learners_pickle_path='drive_data/pickles/base_learner_pickles/',
                                       meta_learner_pickle_path='drive_data/pickles/meta_learner_pickles/')

    RANDOM_MAZE_RESISTANCE = PREDICTION_MODEL.predict(FEATURES_DICT)
    print(f"The prediction resistance of the random maze is: {RANDOM_MAZE_RESISTANCE}")


def generate_desired_maze():
    prediction_model = PredictionModel(base_learners_pickle_path='drive_data/pickles/base_learner_pickles/',
                                       meta_learner_pickle_path='drive_data/pickles/meta_learner_pickles/')

    STEP_SIZE_FACTOR = 0.5
    SIDE_LENGTH = 20

    DESIRED_RESISTANCE = 40
    WIDTH = 0.05
    HEIGHT = 0.05
    FILLET_RADIUS = 0.04

    TARGET_LOC_MODE = "east"  # east or north
    METHOD = "SA"  # TS or SA

    generative_model = GenerativeModel(prediction_model=prediction_model,
                                       desired_resistance=DESIRED_RESISTANCE,
                                       step_size_factor=STEP_SIZE_FACTOR,
                                       width=WIDTH,
                                       height=HEIGHT,
                                       fillet_radius=FILLET_RADIUS,
                                       target_loc_mode=TARGET_LOC_MODE,
                                       method=METHOD,
                                       side_length=SIDE_LENGTH)

    MAZE, FITNESS = generative_model.generate_maze()
    GenerativeModel.pretty_print_maze(MAZE)
    plot_maze(MAZE)

    print(f"Desired resistance: {DESIRED_RESISTANCE}")
    print(f"The resistance of the maze is: {generative_model.predict_resistance(maze=MAZE)}, with fitness: {FITNESS}")

    build_3d_maze(maze=MAZE, step_size_factor=STEP_SIZE_FACTOR, width=WIDTH, height=HEIGHT, fillet_radius=FILLET_RADIUS)


if __name__ == '__main__':
    random_maze()
    # generate_desired_maze()
