import copy
from typing import Optional

import numpy as np

from src.machine_learning import PredictionModel
from src.maze_functions import (complete_maze,
                                random_maze_generator,
                                extract_features,
                                get_coord_list_matrix)
from .simulated_annealing import SA
from .tabu_search import TS


class GenerativeModel:

    def __init__(self,
                 prediction_model: PredictionModel,
                 desired_resistance: float,
                 step_size_factor: float,
                 width: float,
                 height: float,
                 fillet_radius: float,
                 target_loc_mode: str,
                 method: Optional[str],
                 side_length: float,
                 time_limit: float
                 ):
        self.PREDICTION_MODEL = prediction_model
        self.desired_resistance = desired_resistance
        self.step_size_factor = step_size_factor
        self.width = width
        self.height = height
        self.fillet_radius = fillet_radius
        self.target_loc_mode = target_loc_mode
        self.method = method
        self.side_length = side_length
        self.time_limit = time_limit

    def generate_maze(self) -> np.ndarray:

        if self.method == "TS":
            solution = TS(init_method=self.initialization,
                          N_List=[self.N1, self.N2, self.N3],
                          objective_method=self.fitness_function,
                          tabu_size=10,
                          num_neighbors=50,
                          time_limit=self.time_limit,
                          print_iteration=True)

        elif self.method == "SA":
            solution = SA(init_method=self.initialization,
                          N_List=[self.N1, self.N2, self.N3],
                          objective_method=self.fitness_function,
                          time_limit=self.time_limit,
                          print_iteration=True)

        elif self.method is None:
            solution = self.initialization()

        else:
            raise ValueError("method should be either 'TS' or 'SA' or None")

        return solution

    def initialization(self) -> tuple[np.ndarray, float]:
        """
        Initializes the maze by generating a random maze and calculating its fitness.
        :return: random_maze: The randomly generated maze.
        """

        if self.desired_resistance < 10:
            path_finding_mode = "shortest"
        elif self.desired_resistance > 50:
            path_finding_mode = "longest"
        else:
            path_finding_mode = "random"

        random_maze = random_maze_generator(side_length=self.side_length,
                                            target_loc_mode=self.target_loc_mode,
                                            path_finding_mode=path_finding_mode)

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

        fitness = (abs(self.desired_resistance - resistance) / self.desired_resistance) * 100

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

        LOCATIONS_IN_MAZE = get_coord_list_matrix(maze)
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
        LEFT_LOCATIONS = get_coord_list_matrix(DELETED_MAZE)

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
