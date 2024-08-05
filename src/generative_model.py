import copy

import numpy as np

from prediction_model import PredictionModel
from feature_extractor import extract_features, get_coord_list
from maze_generation import complete_maze, random_maze_generator, plot_maze
from tabu_search import TS
from simulated_annealing import SA


class GenerativeModel:

    def __init__(self,
                 prediction_model: PredictionModel,
                 desired_resistance: float,
                 step_size_factor: float,
                 width: float,
                 height: float,
                 fillet_radius: float,
                 target_loc_mode: str,
                 side_length: int = 20
                 ):
        self.PREDICTION_MODEL = prediction_model
        self.desired_resistance = desired_resistance
        self.step_size_factor = step_size_factor
        self.width = width
        self.height = height
        self.fillet_radius = fillet_radius
        self.target_loc_mode = target_loc_mode
        self.side_length = side_length

    def generate_maze(self, method: str = "TS") -> np.ndarray:

        if method == "TS":
            solution = TS(init_method=self.initialization,
                          N_List=[self.N1, self.N2, self.N3],
                          N_func_weights=[0.1, 0.8, 0.1],
                          objective_method=self.fitness_function,
                          tabu_size=10,
                          num_neighbors=50,
                          time_limit=100,
                          print_iteration=True)

        elif method == "SA":
            solution = SA(init_method=self.initialization,
                          N_List=[self.N1, self.N2, self.N3],
                          N_func_weights=[0.1, 0.8, 0.1],
                          objective_method=self.fitness_function,
                          time_limit=100,
                          print_iteration=True)

        else:
            raise ValueError("method should be either 'TS' or 'SA'")

        return solution

    def initialization(self) -> tuple[np.ndarray, float]:
        random_maze = random_maze_generator(side_length=self.side_length,
                                            target_loc_mode=self.target_loc_mode,
                                            path_finding_mode="random")
        fitness, _ = self.fitness_function(random_maze)
        return random_maze, fitness

    def predict_resistance(self,
                           maze: np.ndarray) -> float:
        feature_dict = extract_features(maze=maze,
                                        step_size_factor=self.step_size_factor,
                                        width=self.width,
                                        height=self.height,
                                        fillet_radius=self.fillet_radius)

        resistance = self.PREDICTION_MODEL.predict(data_point_dict=feature_dict)

        return resistance

    def fitness_function(self,
                         maze: np.ndarray) -> tuple[float, bool]:
        resistance = self.predict_resistance(maze=maze)

        fitness = abs(self.desired_resistance - resistance)

        return fitness, True

    def N1(self, current_sol) -> tuple[tuple[np.ndarray, float], tuple[int, str]]:
        new_maze, move = self.neighborhood_function(current_sol, "shortest")
        fitness, _ = self.fitness_function(new_maze)
        return (new_maze, fitness), move

    def N2(self, current_sol) -> tuple[tuple[np.ndarray, float], tuple[int, str]]:
        new_maze, move = self.neighborhood_function(current_sol, "longest")
        fitness, _ = self.fitness_function(new_maze)
        return (new_maze, fitness), move

    def N3(self, current_sol) -> tuple[tuple[np.ndarray, float], tuple[int, str]]:
        new_maze, move = self.neighborhood_function(current_sol, "random")
        fitness, _ = self.fitness_function(new_maze)
        return (new_maze, fitness), move

    @staticmethod
    def neighborhood_function(maze_and_fitness: tuple[np.ndarray, float],
                              path_finding_mode: str
                              ) -> tuple[np.ndarray, tuple[int, str]]:

        maze = copy.deepcopy(maze_and_fitness[0])
        fitness = maze_and_fitness[1]

        LOCATIONS_IN_MAZE = get_coord_list(maze)
        len_maze = len(LOCATIONS_IN_MAZE)

        max_cells_to_delete = int(len_maze // 2)
        if fitness < 2:
            max_cells_to_delete = int(len_maze // 3)
        elif fitness < 1.5:
            max_cells_to_delete = int(len_maze // 4)
        elif fitness < 1:
            max_cells_to_delete = int(len_maze // 5)
        elif fitness < 0.5:
            max_cells_to_delete = int(len_maze // 10)

        # randomly select two locations and delete the path between them
        RANDOM_INDEX_1 = np.random.choice(range(1, len_maze - 1))
        # RANDOM_INDEX_2 can be greater than RANDOM_INDEX_1 by maximum max_cells_to_delete
        RANDOM_INDEX_2 = np.random.choice(range(RANDOM_INDEX_1, min(RANDOM_INDEX_1 + max_cells_to_delete, len_maze)))
        RANDOM_INDICES = np.sort([RANDOM_INDEX_1, RANDOM_INDEX_2])

        DELETED_LOCATIONS_IN_MAZE = LOCATIONS_IN_MAZE[RANDOM_INDICES[0]: RANDOM_INDICES[1]]

        DELETED_MAZE = np.copy(maze)
        for X, Y, V in DELETED_LOCATIONS_IN_MAZE:
            DELETED_MAZE[X, Y] = 0

        # ----------- FILL THE GAP -----------
        LEFT_LOCATIONS = get_coord_list(DELETED_MAZE)

        LAST_LOCATION_BEFORE_EMPTY = LOCATIONS_IN_MAZE[RANDOM_INDICES[0] - 1]
        TARGET_LOCATION = (LOCATIONS_IN_MAZE[RANDOM_INDICES[1]][0], LOCATIONS_IN_MAZE[RANDOM_INDICES[1]][1])
        START_COORDS = [LAST_LOCATION_BEFORE_EMPTY[0], LAST_LOCATION_BEFORE_EMPTY[1],
                        maze[LAST_LOCATION_BEFORE_EMPTY[0], LAST_LOCATION_BEFORE_EMPTY[1]].item()]

        NEW_MAZE = complete_maze(DELETED_MAZE,
                                 START_COORDS,
                                 TARGET_LOCATION,
                                 LEFT_LOCATIONS,
                                 path_finding_mode)

        NUMBER_OF_DELETED_CELLS = len(DELETED_LOCATIONS_IN_MAZE)

        return NEW_MAZE, (NUMBER_OF_DELETED_CELLS, path_finding_mode)

    @staticmethod
    def pretty_print_maze(maze: np.ndarray):
        for row in maze:
            print(" ".join(str(cell).rjust(4) for cell in row))


if __name__ == '__main__':
    PredictionModel = PredictionModel(base_learners_pickle_path='../data/pickles/base_learner_pickles/',
                                      meta_learner_pickle_path='../data/pickles/meta_learner_pickles/')

    DESIRED_RESISTANCE = 30
    WIDTH = 0.10
    HEIGHT = 0.10
    FILLET_RADIUS = 0.10
    GenerativeModel = GenerativeModel(prediction_model=PredictionModel,
                                      desired_resistance=DESIRED_RESISTANCE,
                                      step_size_factor=0.5,
                                      width=WIDTH,
                                      height=HEIGHT,
                                      fillet_radius=FILLET_RADIUS,
                                      target_loc_mode="north",
                                      side_length=20)

    MAZE, FITNESS = GenerativeModel.generate_maze(method="SA")
    GenerativeModel.pretty_print_maze(MAZE)
    plot_maze(MAZE)

    print(f"Desired resistance: {DESIRED_RESISTANCE}")
    print(f"The resistance of the maze is: {GenerativeModel.predict_resistance(maze=MAZE)}, with fitness: {FITNESS}")
