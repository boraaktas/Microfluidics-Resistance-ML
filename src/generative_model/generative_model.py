import copy
from typing import Optional

import numpy as np

from src.machine_learning import PredictionModel
from src.maze_functions import (complete_maze,
                                random_maze_generator,
                                extract_features,
                                get_coord_list_matrix,
                                get_coord_corners_list)
from .simulated_annealing import SA
from .tabu_search import TS


class GenerativeModel:

    def __init__(self,
                 prediction_model: PredictionModel,
                 resistance_bounds_dict: dict,
                 desired_resistance: float,
                 step_size_factor: float,
                 width: float,
                 height: float,
                 fillet_radius: float,
                 target_loc_mode: str,
                 method: Optional[str],
                 side_length: float,
                 time_limit: float,
                 plot_bool: bool = False,
                 print_iteration: bool = False
                 ):
        self.PREDICTION_MODEL = prediction_model
        self.resistance_bounds_dict = resistance_bounds_dict
        self.desired_resistance = desired_resistance
        self.step_size_factor = step_size_factor
        self.width = width
        self.height = height
        self.fillet_radius = fillet_radius
        self.target_loc_mode = target_loc_mode
        self.method = method
        self.side_length = side_length
        self.time_limit = time_limit
        self.plot_bool = plot_bool
        self.print_iteration = print_iteration

        self.check_input_feasibility()

    def check_input_feasibility(self):
        # find the group with given width, height and fillet radius
        selected_group = None
        for comb_value, comb_info in self.resistance_bounds_dict.items():
            if (comb_info["Width"] == self.width
                    and comb_info["Height"] == self.height
                    and comb_info["Fillet_Radius"] == self.fillet_radius):
                selected_group = comb_info
                break

        if selected_group is None:
            raise ValueError("The given width, height and fillet radius combination is not valid.")

        # check if the desired resistance is in the range
        if not selected_group["lb"] <= self.desired_resistance <= selected_group["ub"]:
            msg = (f"The desired resistance should be in the range of "
                   f"[{selected_group['lb']}, {selected_group['ub']}] for"
                   f" the given width ({self.width}), height ({self.height}) and"
                   f" fillet radius ({self.fillet_radius}).")
            raise ValueError(msg)

    def generate_maze(self) -> np.ndarray:

        if self.method == "TS":
            solution = TS(init_method=self.initialization,
                          N_List=[self.N1, self.N2],
                          objective_method=self.fitness_function,
                          tabu_size=10,
                          num_neighbors=50,
                          time_limit=self.time_limit,
                          plot_best_solution=self.plot_bool,
                          print_iteration=self.print_iteration)

        elif self.method == "SA":
            solution = SA(init_method=self.initialization,
                          N_List=[self.N1, self.N2],
                          objective_method=self.fitness_function,
                          time_limit=self.time_limit,
                          plot_best_solution=self.plot_bool,
                          print_iteration=self.print_iteration)

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

        diff_res = self.desired_resistance - resistance

        # if the resistance is higher than the desired resistance, it is more penalized
        if diff_res < 0:
            diff_res *= 1.05

        fitness = (abs(diff_res) / self.desired_resistance) * 100

        return fitness, True

    @staticmethod
    def normalize_weights(weights: list[float]) -> list[float]:
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]

        return normalized_weights

    def N1(self, current_sol) -> tuple[tuple[np.ndarray, float], tuple[int, str]]:
        repair_modes = ["shortest", "longest", "random"]
        repair_modes_weights = [1, 1, 1]

        selection_modes = ["from_lines", "from_corners"]
        selection_modes_weights = [1, 1]

        repair_mode = np.random.choice(repair_modes, p=GenerativeModel.normalize_weights(repair_modes_weights))
        selection_mode = np.random.choice(selection_modes, p=GenerativeModel.normalize_weights(selection_modes_weights))

        new_maze, move = self.destruct_repair_between_two_points(current_sol,
                                                                 selection_mode,
                                                                 repair_mode)
        fitness, _ = self.fitness_function(new_maze)

        return (new_maze, fitness), move

    def N2(self, current_sol) -> tuple[tuple[np.ndarray, float], tuple[int, str]]:
        destruct_modes = ["target", "beginning"]
        destruct_modes_weights = [1, 1]

        repair_modes = ["shortest", "longest", "random"]
        repair_modes_weights = [1, 1, 1]

        selection_modes = ["from_lines", "from_corners"]
        selection_modes_weights = [1, 1]

        destruct_mode = np.random.choice(destruct_modes, p=GenerativeModel.normalize_weights(destruct_modes_weights))
        repair_mode = np.random.choice(repair_modes, p=GenerativeModel.normalize_weights(repair_modes_weights))
        selection_mode = np.random.choice(selection_modes, p=GenerativeModel.normalize_weights(selection_modes_weights))

        new_maze, move = self.destruct_repair_between_point_and_target_or_beginning(current_sol,
                                                                                    selection_mode,
                                                                                    destruct_mode,
                                                                                    repair_mode)
        fitness, _ = self.fitness_function(new_maze)

        return (new_maze, fitness), move

    @staticmethod
    def destruct_repair_between_point_and_target_or_beginning(maze_and_fitness: tuple[np.ndarray, float],
                                                              selection_mode: str,
                                                              destruct_mode: str,  # "target" or "beginning"
                                                              repair_mode: str
                                                              ) -> tuple[np.ndarray, tuple[int, str]]:

        if destruct_mode not in ["target", "beginning"]:
            raise ValueError("destruct_mode should be either 'target' or 'beginning'")

        maze = copy.deepcopy(maze_and_fitness[0])

        LOCATIONS_IN_MAZE = get_coord_list_matrix(maze)
        len_maze = len(LOCATIONS_IN_MAZE)

        first_index = 1 if destruct_mode == "beginning" else len_maze - 1

        # randomly select a location and delete the path between it and the target
        RANDOM_INDEX = GenerativeModel.choose_random_index(maze,
                                                           selection_mode=selection_mode,
                                                           other_chosen_index=first_index
                                                           )

        # ----------- DESTRUCT THE MAZE -----------
        (DELETED_MAZE,
         DELETED_LOCATIONS_IN_MAZE,
         SORTED_INDICES) = GenerativeModel.destruct_maze(maze=maze,
                                                         locations_in_maze=LOCATIONS_IN_MAZE,
                                                         index_1=RANDOM_INDEX,
                                                         index_2=first_index)

        # ----------- REPAIR THE MAZE -----------
        NEW_MAZE = GenerativeModel.repair_maze(deleted_maze=DELETED_MAZE,
                                               old_maze=maze,
                                               locations_in_maze=LOCATIONS_IN_MAZE,
                                               random_indices=SORTED_INDICES,
                                               repair_mode=repair_mode)

        NUMBER_OF_DELETED_CELLS = len(DELETED_LOCATIONS_IN_MAZE)

        return NEW_MAZE, (NUMBER_OF_DELETED_CELLS, repair_mode)

    @staticmethod
    def destruct_repair_between_two_points(maze_and_fitness: tuple[np.ndarray, float],
                                           selection_mode: str,
                                           repair_mode: str
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
        RANDOM_INDEX_1 = GenerativeModel.choose_random_index(maze, selection_mode=selection_mode)
        # RANDOM_INDEX_2 can be greater than RANDOM_INDEX_1 by maximum max_cells_to_delete
        RANDOM_INDEX_2 = GenerativeModel.choose_random_index(maze,
                                                             selection_mode=selection_mode,
                                                             other_chosen_index=RANDOM_INDEX_1,
                                                             max_distance_between_indices=max_cells_to_delete)

        # ----------- DESTRUCT THE MAZE -----------
        (DELETED_MAZE,
         DELETED_LOCATIONS_IN_MAZE,
         SORTED_INDICES) = GenerativeModel.destruct_maze(maze=maze,
                                                         locations_in_maze=LOCATIONS_IN_MAZE,
                                                         index_1=RANDOM_INDEX_1,
                                                         index_2=RANDOM_INDEX_2)

        # ----------- REPAIR THE MAZE -----------
        NEW_MAZE = GenerativeModel.repair_maze(deleted_maze=DELETED_MAZE,
                                               old_maze=maze,
                                               locations_in_maze=LOCATIONS_IN_MAZE,
                                               random_indices=SORTED_INDICES,
                                               repair_mode=repair_mode)

        NUMBER_OF_DELETED_CELLS = len(DELETED_LOCATIONS_IN_MAZE)

        return NEW_MAZE, (NUMBER_OF_DELETED_CELLS, repair_mode)

    @staticmethod
    def choose_random_index(maze: np.ndarray,
                            selection_mode: str,
                            other_chosen_index: int = None,
                            max_distance_between_indices: int = 30,
                            ) -> int:

        LOCATIONS_IN_MAZE = get_coord_list_matrix(maze)
        CORNERS_IN_MAZE = get_coord_corners_list(LOCATIONS_IN_MAZE)

        if selection_mode == "from_lines":
            locations_in_maze_weights = []
            for i in range(len(LOCATIONS_IN_MAZE)):
                if i == 0 or i == len(LOCATIONS_IN_MAZE) - 1:
                    continue

                if other_chosen_index is None:
                    locations_in_maze_weights.append(1)
                else:
                    if abs(i - other_chosen_index) > max_distance_between_indices:
                        locations_in_maze_weights.append(0)
                    else:
                        locations_in_maze_weights.append(1)

            locations_in_maze_weights = GenerativeModel.normalize_weights(locations_in_maze_weights)

            if sum(locations_in_maze_weights) == 0:
                random_index = np.random.choice(range(1, len(LOCATIONS_IN_MAZE) - 1))
            else:
                random_index = np.random.choice(range(1, len(LOCATIONS_IN_MAZE) - 1), p=locations_in_maze_weights)

        elif selection_mode == "from_corners":
            locations_in_maze_weights = []
            for i in range(len(LOCATIONS_IN_MAZE)):
                if i == 0 or i == len(LOCATIONS_IN_MAZE) - 1:
                    continue

                loc = LOCATIONS_IN_MAZE[i]
                if loc in CORNERS_IN_MAZE:
                    if other_chosen_index is None:
                        locations_in_maze_weights.append(1)
                    else:
                        if abs(i - other_chosen_index) > max_distance_between_indices:
                            locations_in_maze_weights.append(0)
                        else:
                            locations_in_maze_weights.append(1)
                else:
                    locations_in_maze_weights.append(0)

            locations_in_maze_weights = GenerativeModel.normalize_weights(locations_in_maze_weights)

            if sum(locations_in_maze_weights) == 0:
                random_index = np.random.choice([1, len(LOCATIONS_IN_MAZE) - 1])
            else:
                random_index = np.random.choice(range(1, len(LOCATIONS_IN_MAZE) - 1), p=locations_in_maze_weights)

        else:
            raise ValueError("selection_mode should be either 'from_lines' or 'from_corners'")

        return random_index

    @staticmethod
    def destruct_maze(maze: np.ndarray,
                      locations_in_maze: list[list[int]],
                      index_1: int,
                      index_2: int) -> tuple[np.ndarray, list[list[int]], np.ndarray]:

        sorted_indices = np.sort([index_1, index_2])

        deleted_locations_in_maze = locations_in_maze[sorted_indices[0]: sorted_indices[1]]

        deleted_maze = np.copy(maze)
        for X, Y, V in deleted_locations_in_maze:
            deleted_maze[X, Y] = 0

        return deleted_maze, deleted_locations_in_maze, sorted_indices

    @staticmethod
    def repair_maze(deleted_maze: np.ndarray,
                    old_maze: np.ndarray,
                    locations_in_maze: list[list[int]],
                    random_indices: np.ndarray,
                    repair_mode: str) -> np.ndarray:

        LEFT_LOCATIONS = get_coord_list_matrix(deleted_maze)

        LAST_LOCATION_BEFORE_EMPTY = locations_in_maze[random_indices[0] - 1]

        TARGET_LOCATION = (locations_in_maze[random_indices[1]][0], locations_in_maze[random_indices[1]][1])
        START_COORDS = [LAST_LOCATION_BEFORE_EMPTY[0], LAST_LOCATION_BEFORE_EMPTY[1],
                        old_maze[LAST_LOCATION_BEFORE_EMPTY[0], LAST_LOCATION_BEFORE_EMPTY[1]].item()]

        NEW_MAZE = complete_maze(deleted_maze,
                                 START_COORDS,
                                 TARGET_LOCATION,
                                 LEFT_LOCATIONS,
                                 repair_mode)

        return NEW_MAZE

    @staticmethod
    def pretty_print_maze(maze: np.ndarray):
        for row in maze:
            print(" ".join(str(cell).rjust(4) for cell in row))
