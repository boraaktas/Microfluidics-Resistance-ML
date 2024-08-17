import math
import random
import time
from datetime import datetime
from typing import Callable, Union, Optional

from matplotlib import pyplot as plt

from src.maze_functions import plot_maze


def TS(init_method: Callable,
       N_List: list[Callable],
       objective_method: Callable,
       init_sol: Optional[list] = None,
       tabu_size: int = 15,
       num_neighbors: int = 100,
       N_func_weights: Union[str, list[Union[float, int]]] = 'uniform',
       reaction_factor: float = 1,
       time_limit: float = 1000,
       timer_starts_after_initialization: bool = False,
       ITER: int = 1000000,
       benchmark_cost: float = 0,
       benchmark_opt_bool: bool = False,
       threshold_gap: float = 0.01,
       print_iteration: bool = False,
       print_results: bool = False,
       plot_results: bool = False,
       plot_best_solution: bool = False,
       save_results: bool = False,
       save_results_path: str = '',
       save_results_file_name: str = ''):
    """
    Tabu Search Algorithm
    """

    check_parameters(N_List=N_List,
                     N_func_weights=N_func_weights,
                     reaction_factor=reaction_factor,
                     save_results=save_results,
                     save_results_path=save_results_path,
                     save_results_file_name=save_results_file_name)

    start_time: float = time.time()  # start time of the algorithm

    if init_sol is not None:
        S_initial: list = init_sol
    else:
        # Initialize the solution and its cost
        S_initial: list = init_method()

    if timer_starts_after_initialization:
        start_time = time.time()

    objective_return = S_initial
    cost_initial: float = objective_return[1]
    feasibility_initial: bool = True

    # Initialize the current solution and its cost
    S_current = (S_initial[0].copy(), cost_initial)
    cost_current: float = cost_initial
    feasibility_current: bool = feasibility_initial

    # Initialize the best solution and its cost as the current solution and its cost
    S_best = (S_current[0].copy(), cost_current)
    cost_best: float = cost_current
    feasibility_best: bool = feasibility_current
    best_time: float = start_time
    best_iter: int = 0

    # Keep track of all the solutions found so far
    solutions: list = [S_current]
    solution_costs: list[float] = [cost_current]

    # Number of neighbor functions
    no_functions: int = len(N_List)

    # Initialize the probabilities of choosing each neighbor function
    N_func_probs: list[float] = [0 for _ in range(no_functions)]

    if isinstance(N_func_weights, str) and N_func_weights.lower() == 'uniform':
        N_func_probs = [1 / no_functions for _ in range(no_functions)]
    elif isinstance(N_func_weights, list):
        N_func_probs = [p / sum(N_func_weights) for p in N_func_weights]

    # Keep track of the number of times each neighbor function was the best neighbor selected in each iteration
    N_func_selected: list[int] = [0 for _ in range(no_functions)]
    # Keep track of the number of times each neighbor function improved the best solution
    N_func_improves: list[int] = [0 for _ in range(no_functions)]

    # Keep track of the gap between the current solution found so far and the benchmark cost
    gap: float = abs(cost_current - benchmark_cost) / \
                 benchmark_cost if benchmark_cost != 0 else math.inf
    threshold_gap_time: Optional[float] = None
    threshold_gap_iter: Optional[int] = None
    if gap <= threshold_gap:
        threshold_gap_time = time.time() - start_time
        threshold_gap_iter = 0

    iteration = 0

    # -------------------------------------------------- TABU SEARCH ---------------------------------------------------

    # Initialize the tabu list
    tabu_list: list = []

    # Keep track of the number of times a tabu solution was chased and selected
    number_of_chased_tabu: int = 0

    # Keep track of the number of times a tabu solution met the aspiration criteria and was selected
    number_of_selected_tabu: int = 0

    while not check_termination_conditions(iteration=iteration,
                                           ITER=ITER,
                                           start_time=start_time,
                                           time_limit=time_limit,
                                           benchmark_opt_bool=benchmark_opt_bool,
                                           gap=gap,
                                           threshold_gap=threshold_gap,
                                           cost_best=cost_best
                                           ):

        # Generate all neighbors of the current solution and their moves (the moves are stored in a dictionary)
        # The dictionary is of the form {neighbor: move} where move is a tuple of the form (i, j)
        # neighbors are created by applying the neighbor functions according to function probabilities
        neighbors, moves = generate_neighbors(
            N_List, N_func_probs, S_current, num_neighbors)

        costs: list[float] = []
        feasibilities: list[bool] = []
        for neighbor in neighbors:
            neighbor_objective, neighbor_feasibility = neighbor[1], True
            costs.append(neighbor_objective)
            feasibilities.append(neighbor_feasibility)

        # sort the neighbors and their costs in ascending order
        # neighbors_and_moves structure: [(cost, feasibility, neighbor, move), ...]
        # move is a tuple of the form ('function_name', move_tuple)
        neighbors_and_moves = sorted(zip(costs, feasibilities,
                                         neighbors, moves),
                                     key=lambda x: x[0])

        best_neighbor_found: bool = False
        best_neighbor_cost: float = neighbors_and_moves[0][0]
        best_neighbor_feasibility: bool = neighbors_and_moves[0][1]
        best_neighbor_solution: list = neighbors_and_moves[0][2]
        best_neighbor_move: tuple = neighbors_and_moves[0][3]

        index = 0
        while not best_neighbor_found and index < len(neighbors_and_moves):
            best_neighbor_cost = neighbors_and_moves[index][0]
            best_neighbor_feasibility = neighbors_and_moves[index][1]
            best_neighbor_solution = neighbors_and_moves[index][2]
            best_neighbor_move = neighbors_and_moves[index][3]

            meets_aspiration_criteria: bool = False
            in_tabu_list = check_tabu_list(best_neighbor_move,
                                           tabu_list)

            if in_tabu_list:
                meets_aspiration_criteria, tabu_list = check_aspiration_criteria(best_neighbor_move,
                                                                                 best_neighbor_cost,
                                                                                 cost_best,
                                                                                 tabu_list)
                number_of_chased_tabu += 1

            if (not in_tabu_list) or meets_aspiration_criteria:
                best_neighbor_found = True
                if in_tabu_list:
                    number_of_selected_tabu += 1

            index += 1

        # if the best neighbor is not found, continue with the current solution
        if not best_neighbor_found:
            best_neighbor_cost = cost_current
            best_neighbor_feasibility = feasibility_current
            best_neighbor_solution = (S_current[0].copy(), cost_current)
            best_neighbor_move = ('None', ())

        # find the index of the best neighbor function
        best_neighbor_move_func_name = best_neighbor_move[0]
        best_neighbor_move_func_index = list(
            map(lambda x: x.__name__, N_List)).index(best_neighbor_move_func_name)
        N_func_selected[best_neighbor_move_func_index] += 1

        # Update the probabilities of choosing each neighbor function
        if reaction_factor != 1:
            # Update the probabilities of choosing each neighbor function
            N_func_probs = change_function_probabilities_2(func_probs=N_func_probs,
                                                           best_move_func_index=best_neighbor_move_func_index,
                                                           selected_solution_obj=best_neighbor_cost,
                                                           best_solution_obj=cost_best,
                                                           current_solution_obj=cost_current,
                                                           react_factor=reaction_factor)

        # Update the current solution and its cost
        S_current = (best_neighbor_solution[0].copy(), best_neighbor_cost)
        cost_current = best_neighbor_cost

        # Update the gap between the current solution found so far and the benchmark cost
        gap = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
        if gap <= threshold_gap and threshold_gap_time is None and threshold_gap_iter is None:
            threshold_gap_time = time.time() - start_time
            threshold_gap_iter = iteration

        # Add the move that was made to the tabu list
        tabu_list.append(best_neighbor_move)

        # Remove the oldest move from the tabu list if the tabu list is full
        if len(tabu_list) == tabu_size:
            tabu_list.pop(0)

        # Add the current solution to the list of solutions
        solutions.append(S_current)
        solution_costs.append(cost_current)

        # Update the iteration counter
        iteration += 1

        # Update the best solution and its cost if the current solution is better than the best solution found so far
        if cost_current < cost_best:
            S_best = (S_current[0].copy(), cost_current)
            cost_best = cost_current
            feasibility_best = best_neighbor_feasibility
            best_time = time.time() - start_time
            best_iter = iteration

            # track the number of times each neighbor function improved the best solution
            N_func_improves[best_neighbor_move_func_index] += 1

            if plot_best_solution:
                plot_maze(S_best[0])

        if print_iteration:
            # print the current iteration information""
            print("\n--------------------------------------------------")
            print("Iteration: ", iteration)
            print("Time elapsed: ", time.time() - start_time)
            print("Old solution: ", solutions[-2])
            print("Move: ", best_neighbor_move)
            print("Selected (current) solution: ", S_current)
            print("Current cost: ", cost_current)
            print("Feasibility: ", feasibility_current)
            print("Tabu list: ", tabu_list)
            print("Probs of choosing each neighbor function: ", N_func_probs)
            print("\nBest solution: ", S_best)
            print("Best cost: ", cost_best)
            print("Best Feasibility: ", feasibility_best)
            print("Best gap: ", abs(cost_best - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf)
            print("Best time: ", best_time)
            print("Best iteration: ", best_iter)
            print("--------------------------------------------------\n")

    # -------------------------------------------------- TABU SEARCH ---------------------------------------------------

    # time elapsed since the start of the algorithm
    time_elapsed = time.time() - start_time

    if print_results:
        print("-------------------- RESULTS --------------------")
        print("Run parameters:")
        print("Tabu size:", tabu_size)
        print("Number of neighbors:", num_neighbors)
        print("Reaction factor:", reaction_factor)

        print("\nClassic results:")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("Feasibility: ", feasibility_best)
        print("Iterations:", iteration)
        print("Time elapsed:", time_elapsed)
        print("Best time:", best_time)
        print("Best iteration:", best_iter)
        print("Initial method:", init_method.__name__)
        print("Initial solution:", S_initial)
        print("Initial cost:", cost_initial)
        print("Initial feasibility:", feasibility_initial)
        print("Benchmark cost:", benchmark_cost)
        print("Threshold gap:", threshold_gap)
        print("Threshold gap time:", threshold_gap_time)
        print("Threshold gap iteration:", threshold_gap_iter)
        print("Time limit:", time_limit)
        print("Iteration limit:", ITER)

        print("\nAlgorithm specific results:")
        print("Number of times a tabu solution was chased and selected:",
              number_of_chased_tabu)
        print("Number of times a tabu solution met the aspiration criteria and was selected:",
              number_of_selected_tabu)

        print("\nOther results:")
        print("Number of times each neighbor function was selected:", N_func_selected)
        print("Number of times each neighbor function improved the best solution:", N_func_improves)
        print("Probability of choosing each neighbor function at the end:", N_func_probs)

        print("\n--------------------------------------------------")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("--------------------------------------------------\n")

    if plot_results:
        plt.plot(solution_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Current cost vs Iteration")
        plt.axhline(y=cost_best, color='r', linestyle='-')
        plt.show()

    run_params = {'tabu_size': tabu_size,
                  'num_neighbors': num_neighbors,
                  'reaction_factor': reaction_factor}

    classic_results = {'best_solution': S_best,
                       'best_cost': cost_best,
                       'feasibility': feasibility_best,
                       'iterations': iteration,
                       'time_elapsed': time_elapsed,
                       'best_time': best_time,
                       'best_iter': best_iter,
                       'init_method': init_method.__name__,
                       'initial_solution': S_initial,
                       'initial_cost': cost_initial,
                       'initial_feasibility': feasibility_initial,
                       'benchmark_cost': benchmark_cost,
                       'benchmark_opt_bool': benchmark_opt_bool,
                       'threshold_gap': threshold_gap,
                       'threshold_gap_time': threshold_gap_time,
                       'threshold_gap_iter': threshold_gap_iter,
                       'time_limit': time_limit,
                       'iter_limit': ITER,
                       'date_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    alg_specific_results = {'number_of_chased_tabu': number_of_chased_tabu,
                            'number_of_selected_tabu': number_of_selected_tabu}

    if save_results:
        '''Holder.save_results(algorithm='TS',
                            algorithm_params=run_params,
                            classic_results=classic_results,
                            algorithm_results=alg_specific_results,
                            path=save_results_path,
                            file_name=save_results_file_name)'''
        pass

    return S_best


def check_parameters(N_List: list[Callable],
                     N_func_weights: Union[str,
                     list[Union[float, int]]] = 'uniform',
                     reaction_factor: float = 1,
                     save_results: bool = False,
                     save_results_path: str = '',
                     save_results_file_name: str = '') -> None:
    """
    Check if the parameters of the Tabu Search algorithm are valid
    """

    no_functions: int = len(N_List)

    if no_functions == 0:
        # If the list of neighbor functions is empty, raise an error
        msg = "The list of neighbor functions is empty."
        raise ValueError(msg)
    elif no_functions != len(N_func_weights) and isinstance(N_func_weights, list):
        # If the number of neighbor functions and the number of probabilities of choosing each neighbor function
        # are not the same, raise an error
        msg = "The number of neighbor functions and the number of probabilities of choosing each neighbor function " \
              "are not the same."
        raise ValueError(msg)

    if isinstance(N_func_weights, str):
        # If the probability distribution of choosing each neighbor function is not a list or 'uniform', raise an error
        if N_func_weights.lower() != 'uniform':
            msg = "The probability distribution of choosing each neighbor function is not valid." \
                  "It should be either 'uniform' or a list of weights."
            raise ValueError(msg)
    elif isinstance(N_func_weights, list):
        # check if all the elements in the list of weights are positive and integers or floats
        for p in N_func_weights:
            if not isinstance(p, (int, float)) or p < 0:
                msg = "The probability distribution of choosing each neighbor function is not valid." \
                      "It should be either 'uniform' or a list of positive weights."
                raise ValueError(msg)

    if reaction_factor < 0 or reaction_factor > 1:
        # If the reaction factor is not between 0 and 1, raise an error
        msg = "The reaction factor should be between 0 and 1."
        raise ValueError(msg)

    if save_results:
        if save_results_path == '':
            msg = "The path to save the results is empty."
            raise ValueError(msg)
        if save_results_file_name == '':
            msg = "The file name to save the results is empty."
            raise ValueError(msg)


def check_termination_conditions(iteration: int,
                                 ITER: int,
                                 start_time: float,
                                 time_limit: float,
                                 benchmark_opt_bool: bool,
                                 gap: float,
                                 threshold_gap: float,
                                 cost_best: float
                                 ) -> bool:
    """
    Check if the termination conditions are met.
    The termination conditions are:
        - The number of iterations exceeds the maximum number of iterations
        - The elapsed time exceeds the maximum time
        - The gap between the current solution found so far and the benchmark cost is less than the threshold gap
        - The benchmark optimal cost is reached
        ...

    Args:
        iteration (int): number of iterations
        ITER (int): maximum number of iterations
        start_time (float): time at which the algorithm started
        time_limit (float): maximum time allowed for the algorithm to run
        benchmark_opt_bool (bool): True if benchmark cost is also optimal, False otherwise
        gap (float): gap between the current solution found so far and the benchmark cost
        threshold_gap (float): threshold gap between the current solution found so far and the benchmark cost

    Returns:
        bool: True if the termination conditions are met, False otherwise
    """

    if iteration >= ITER:
        return True

    elapsed_time = time.time() - start_time
    if elapsed_time >= time_limit:
        return True

    # Check if the algorithm has reached the benchmark optimal cost
    if benchmark_opt_bool and gap == 0:
        return True

    if gap <= threshold_gap:
        return True

    if cost_best <= 0.5:
        return True

    return False


def generate_neighbors(N_List, N_func_probs, S_current, num_neighbors):
    """
    Generate neighbors of the current solution and their moves (the moves are stored in a dictionary)
    by applying the neighbor function from the list of neighbor functions to the current solution num_neighbors times

    Args:
        N_List (list): list of neighbor functions
        N_func_probs (list): list of probabilities of choosing each neighbor function
        S_current (list): current solution
        num_neighbors (int): number of neighbors to generate

    Returns:
        neighbors (list): list of neighbors of the current solution
        moves (list): list of moves that were made to get to the neighbors in the neighbors list with the same order
    """

    # neighbors is a dictionary of the form {neighbor: move}
    neighbors = []
    moves = []

    for i in range(num_neighbors):
        # Choose a random neighbor function from the list
        rand = random.choices(range(len(N_List)), weights=N_func_probs)[0]
        neighbor_func = N_List[rand]

        neighbor, move = neighbor_func(S_current)

        # add the neighbor and its move to the lists
        # neighbors is a list of list
        neighbors.append(neighbor)

        # moves is a list of tuples of the form ('function_name', move_tuple)
        moves.append(tuple((neighbor_func.__name__, move)))

    return neighbors, moves


def check_tabu_list(move_tuple, tabu_list):
    """
    Check if the move is in the tabu list

    Args:
        move_tuple (tuple): move that was made to get to the solution
        tabu_list (list): list of solutions that are tabu

    Returns:
        bool: True if the move is in the tabu list, False otherwise
    """

    move_func = move_tuple[0]
    move = move_tuple[1]

    for tabu in tabu_list:
        # Check for first move function name
        if tabu[0] == move_func:
            # If the move function name is the same, check for the move
            if tabu[1] == move:
                return True

    return False


def change_function_probabilities_1(N_List: list[Callable],
                                    N_func_probs: list[float],
                                    winner_index: int,
                                    prize: float
                                    ) -> list[float]:
    # increase the probability of the winner function
    N_func_probs[winner_index] += prize

    # rearrange the probabilities to make sure that the sum of the probabilities is 1
    N_func_probs = [p / sum(N_func_probs) for p in N_func_probs]

    #
    if N_func_probs[winner_index] > 0.5:
        N_func_probs = [1 / len(N_List) for _ in range(len(N_List))]

    return N_func_probs


def change_function_probabilities_2(func_probs: list[float],
                                    best_move_func_index: int,
                                    selected_solution_obj: float,
                                    best_solution_obj: float,
                                    current_solution_obj: float,
                                    react_factor: float) -> list[float]:
    """
    Change the probabilities of choosing each neighbor function based on the reaction factor and the prize
    from: https://doi.org/10.1016/j.trc.2019.02.018

    Args:
        func_probs (list): list of probabilities of choosing each neighbor function
        best_move_func_index (int): index of the best neighbor function
        selected_solution_obj (float): objective value of the selected solution
        best_solution_obj (float): objective value of the best solution found so far
        current_solution_obj (float): objective value of the current solution
        react_factor (float): reaction factor that is the weight of the old probabilities of each function
    Returns:
        N_func_probs (list): list of probabilities of choosing each neighbor function
    """

    # Calculate the prize
    if selected_solution_obj > best_solution_obj:
        prize = 0.001
    elif selected_solution_obj > current_solution_obj:
        prize = 0.0001
    else:
        prize = 0.0

    # update the weights of the neighbor functions
    for j in range(len(func_probs)):
        if j == best_move_func_index:
            func_probs[j] = react_factor * \
                            func_probs[j] + prize * (1 - react_factor)
        else:
            func_probs[j] = react_factor * func_probs[j]

    # rearrange the probabilities to make sure that the sum of the probabilities is 1
    func_probs = [prob / sum(func_probs) for prob in func_probs]

    return func_probs


def check_aspiration_criteria(move, neighbor_cost, best_cost, tabu_list) -> tuple[bool, list]:
    """
    Check if the solution is in the tabu list and if it is,
    check if it is better than the best solution in the tabu list

    Args:
        move (tuple): move that was made to get to the neighbor solution
        neighbor_cost (float): cost of the neighbor solution
        best_cost (float): cost of the best solution found so far
        tabu_list (list): list of solutions that are tabu

    Returns: bool: True if the solution is in the tabu list,
                        and it is better than the best solution in the tabu list,
                    False otherwise
    """

    # Check if the neighbor solution that is in the tabu list is better than the best solution found so far
    if neighbor_cost < best_cost:
        """
        print('Aspiration criteria met')
        print('Move selected from the tabu list: ', move)
        """
        # update the tabu list
        tabu_list.remove(move)
        return True, tabu_list

    return False, tabu_list
