import copy
import random
import time
from typing import Callable
from typing import Optional

import numpy as np

from src.machine_learning import PredictionModel
from src.maze_functions import plot_maze, pretty_print_maze
from src.maze_functions import (random_maze_generator,
                                extract_features,
                                get_coord_list_matrix,
                                get_coord_corners_list,
                                destruct_maze,
                                repair_maze)


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
                 iteration_limit: int,
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
        self.iteration_limit = iteration_limit
        self.plot_bool = plot_bool
        self.print_iteration = print_iteration

        self.selected_group = self.check_input_feasibility()

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
        if not selected_group["lb"] * 0.999 <= self.desired_resistance <= selected_group["ub"] * 1.001:
            msg = (f"The desired resistance should be in the range of "
                   f"[{selected_group['lb']}, {selected_group['ub']}] for"
                   f" the given width ({self.width}), height ({self.height}) and"
                   f" fillet radius ({self.fillet_radius}). The given desired resistance is {self.desired_resistance}.")
            raise ValueError(msg)

        return selected_group

    def generate_maze(self):

        maze_and_fitness, costs = GenerativeModel.tabu_search(init_method=self.initialization,
                                                  neighbor_func_list=[self.N1, self.N2],
                                                  tabu_size=10,
                                                  neighborhood_size=50,
                                                  time_limit=self.time_limit,
                                                  iter_limit=self.iteration_limit,
                                                  plot_best_solution=self.plot_bool,
                                                  print_iteration=self.print_iteration,
                                                  print_results=True)
        maze = maze_and_fitness[0]

        return maze, costs

    def initialization(self) -> tuple[np.ndarray, float]:
        """
        Initializes the maze by generating a random maze and calculating its fitness.
        :return: random_maze: The randomly generated maze.
        """
        path_finding_mode = "random"

        # if desired resistance is greater than lower bound by 20%, set the path finding mode to "shortest"
        if self.desired_resistance <= self.selected_group["lb"] * (1 + 0.2):
            path_finding_mode = "shortest"

        random_maze = random_maze_generator(side_length=self.side_length,
                                            target_loc_mode=self.target_loc_mode,
                                            path_finding_mode=path_finding_mode)

        fitness = self.fitness_function(random_maze)
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
                         maze: np.ndarray) -> float:
        resistance = self.predict_resistance(maze=maze)

        diff_res = self.desired_resistance - resistance

        # if the resistance is higher than the desired resistance, it is more penalized
        if diff_res < 0:
            diff_res *= 1.05

        fitness = (abs(diff_res) / self.desired_resistance) * 100
        # fitness = abs(diff_res)

        return fitness

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
        fitness = self.fitness_function(new_maze)

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
        fitness = self.fitness_function(new_maze)

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
        DELETED_MAZE, DELETED_LOCATIONS_IN_MAZE = destruct_maze(maze=maze,
                                                                locations_in_maze=LOCATIONS_IN_MAZE,
                                                                index_1=RANDOM_INDEX,
                                                                index_2=first_index)

        SORTED_INDICES = np.sort([RANDOM_INDEX, first_index])

        # ----------- REPAIR THE MAZE -----------
        NEW_MAZE = repair_maze(deleted_maze=DELETED_MAZE,
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
        DELETED_MAZE, DELETED_LOCATIONS_IN_MAZE = destruct_maze(maze=maze,
                                                                locations_in_maze=LOCATIONS_IN_MAZE,
                                                                index_1=RANDOM_INDEX_1,
                                                                index_2=RANDOM_INDEX_2)

        SORTED_INDICES = np.sort([RANDOM_INDEX_1, RANDOM_INDEX_2])

        # ----------- REPAIR THE MAZE -----------
        NEW_MAZE = repair_maze(deleted_maze=DELETED_MAZE,
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

            if sum(locations_in_maze_weights) == 0:
                random_index = np.random.choice(range(1, len(LOCATIONS_IN_MAZE) - 1))
            else:
                locations_in_maze_weights = GenerativeModel.normalize_weights(locations_in_maze_weights)
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

            if sum(locations_in_maze_weights) == 0:
                random_index = np.random.choice([1, len(LOCATIONS_IN_MAZE) - 1])
            else:
                locations_in_maze_weights = GenerativeModel.normalize_weights(locations_in_maze_weights)
                random_index = np.random.choice(range(1, len(LOCATIONS_IN_MAZE) - 1), p=locations_in_maze_weights)

        else:
            raise ValueError("selection_mode should be either 'from_lines' or 'from_corners'")

        return random_index

    @staticmethod
    def tabu_search(init_method: Callable,
                    neighbor_func_list: list[Callable],
                    tabu_size: int = 15,
                    neighborhood_size: int = 100,
                    time_limit: float = 1000,
                    iter_limit: int = 1000000,
                    print_iteration: bool = False,
                    print_results: bool = False,
                    plot_best_solution: bool = False):

        # Initialize
        S_initial: tuple[list, float] = init_method()
        start_time = time.time()

        cost_initial: float = S_initial[1]

        # Current and best solutions
        S_current = (S_initial[0].copy(), cost_initial)
        cost_current: float = cost_initial

        S_best = (S_current[0].copy(), cost_current)
        cost_best: float = cost_current
        best_time: float = start_time
        best_iter: int = 0

        solutions: list = [S_current]
        solution_costs: list[float] = [cost_current]

        # Tabu list
        tabu_list: list = []

        iteration = 0
        while True:
            # Check termination conditions
            if iteration >= iter_limit:
                break
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
            if cost_best <= 0.5:
                break

            # Generate neighbors
            neighbors = []
            moves = []
            for _ in range(neighborhood_size):
                rand_idx = random.choices(range(len(neighbor_func_list)),
                                          weights=[1 / len(neighbor_func_list)] * len(neighbor_func_list))[0]
                neighbor_func = neighbor_func_list[rand_idx]
                neighbor, move = neighbor_func(S_current)
                neighbors.append(neighbor)
                moves.append((neighbor_func.__name__, move))

            costs: list[float] = []
            for neighbor in neighbors:
                neighbor_objective = neighbor[1]
                costs.append(neighbor_objective)

            neighbors_and_moves = sorted(zip(costs, neighbors, moves),
                                         key=lambda x: x[0])

            best_neighbor_found: bool = False
            best_neighbor_cost: float = neighbors_and_moves[0][0]
            best_neighbor_solution = neighbors_and_moves[0][1]
            best_neighbor_move: tuple = neighbors_and_moves[0][2]

            # Find best neighbor that is not tabu or satisfies aspiration
            index = 0
            while not best_neighbor_found and index < len(neighbors_and_moves):
                c_cost = neighbors_and_moves[index][0]
                c_sol = neighbors_and_moves[index][1]
                c_mv = neighbors_and_moves[index][2]

                # Check if move is tabu
                in_tabu_list = False
                for t in tabu_list:
                    if t[0] == c_mv[0] and t[1] == c_mv[1]:
                        in_tabu_list = True
                        break

                meets_aspiration_criteria = False
                if in_tabu_list:
                    # Aspiration criteria
                    if c_cost < cost_best:
                        # Remove from tabu
                        if c_mv in tabu_list:
                            tabu_list.remove(c_mv)
                        meets_aspiration_criteria = True

                if (not in_tabu_list) or meets_aspiration_criteria:
                    best_neighbor_found = True
                    best_neighbor_cost = c_cost
                    best_neighbor_solution = c_sol
                    best_neighbor_move = c_mv

                index += 1

            # If no best neighbor found, stick to current
            if not best_neighbor_found:
                best_neighbor_cost = cost_current
                best_neighbor_solution = (S_current[0].copy(), cost_current)
                best_neighbor_move = ('None', ())

            S_current = (best_neighbor_solution[0].copy(), best_neighbor_cost)
            cost_current = best_neighbor_cost

            # Update tabu list
            tabu_list.append(best_neighbor_move)
            if len(tabu_list) == tabu_size:
                tabu_list.pop(0)

            solutions.append(S_current)
            solution_costs.append(cost_current)

            # Update best solution
            if cost_current < cost_best:
                S_best = (S_current[0].copy(), cost_current)
                cost_best = cost_current
                best_time = time.time() - start_time
                best_iter = iteration

                if plot_best_solution:
                    plot_maze(S_best[0])

            if print_iteration:
                print("\n--------------------------------------------------")
                print(f"Time elapsed: {elapsed_time:.2f}/{time_limit}")
                print(f"Iteration: {iteration}/{iter_limit}")
                print(f"Current cost: {cost_current:.4f}")
                print("Current solution:")
                pretty_print_maze(S_current[0])
                print(f"Best cost: {cost_best:.4f}")
                print("Best solution:")
                pretty_print_maze(S_best[0])
                print("--------------------------------------------------\n")

            iteration += 1

        time_elapsed = time.time() - start_time

        if print_results:
            print("-------------------- RESULTS --------------------")
            print("Tabu size:", tabu_size)
            print("Number of neighbors:", neighborhood_size)
            print(f"Time elapsed: {time_elapsed:.2f}/{time_limit}")
            print(f"Iterations: {iteration}/{iter_limit}")
            print(f"Best time: {best_time:.2f}")
            print("Best iteration:", best_iter)
            print(f"Best cost: {cost_best:.4f}")
            print("Best solution:")
            pretty_print_maze(S_best[0])
            print("--------------------------------------------------\n")

        return S_best, solution_costs
