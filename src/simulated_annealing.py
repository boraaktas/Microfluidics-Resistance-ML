import math
import random
import time
from datetime import datetime
from typing import Callable, Union, Optional

from matplotlib import pyplot as plt

from maze_generation import plot_maze


def SA(init_method: Callable,
       N_List: list[Callable],
       objective_method: Callable,
       init_sol: Optional[list] = None,
       init_temp: float = 0.5,
       cooling_rate: float = 0.99,
       N_func_weights: Union[str, list[float | int]] = 'uniform',
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
       save_results: bool = False,
       save_results_path: str = '',
       save_results_file_name: str = '') -> list:
    """
    Simulated Annealing Algorithm
    """

    # Check the parameters
    check_parameters(N_List=N_List,
                     init_temp=init_temp,
                     cooling_rate=cooling_rate,
                     N_func_weights=N_func_weights,
                     save_results=save_results,
                     save_results_path=save_results_path,
                     save_results_file_name=save_results_file_name)

    start_time: float = time.time()  # start time of the algorithm

    if init_sol is not None:
        S_initial: list = init_sol.copy()
    else:
        # Initialize the solution and its cost
        S_initial: list = init_method()

    if timer_starts_after_initialization:
        start_time = time.time()

    cost_initial, feasibility_initial = S_initial[1], True
    plot_maze(S_initial[0])

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

    # Keep track of the gap between the current solution found so far and the benchmark cost
    gap: float = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
    threshold_gap_time: Optional[float] = None
    threshold_gap_iter: Optional[int] = None
    if gap <= threshold_gap:
        threshold_gap_time = time.time() - start_time
        threshold_gap_iter = 0

    iteration: int = 0  # number of iterations

    # ----------------------------------------------- SIMULATED ANNEALING ----------------------------------------------

    T_0: float = init_temp  # initial temperature
    alpha: float = cooling_rate  # cooling rate

    T: float = T_0  # current temperature
    T_b: float = T_0  # best temperature
    T_max: float = T_0  # maximum temperature

    Len: int = 0  # inner loop iteration number

    # TODO:
    max_Len = 100  # maximum number of iterations for inner loop
    rollback_Len = 5  # number of iterations for inner loop when
    threshold_temp = 0.01  # threshold temperature (if the temperature is too low, reset temperature to max(T_b, T_max))

    # outer loop for the SA algorithm checking the termination conditions
    while not check_termination_conditions(iteration=iteration,
                                           ITER=ITER,
                                           start_time=start_time,
                                           time_limit=time_limit,
                                           benchmark_opt_bool=benchmark_opt_bool,
                                           gap=gap,
                                           threshold_gap=threshold_gap,
                                           cost_best=cost_best
                                           ):

        Len = min(Len + 2, max_Len)  # TODO: Why +2?

        # inner loop for the SA algorithm (temperature)
        for i in range(Len):

            # Choose a random neighbor function from the list
            rand = random.choices(range(no_functions), weights=N_func_probs)[0]
            n_funk = N_List[rand]

            S_prime = n_funk((S_current[0].copy(), S_current[1]))[0]
            cost_prime, feasibility_prime = S_prime[1], True

            if cost_prime < cost_current:
                S_current = (S_prime[0].copy(), cost_prime)
                cost_current = cost_prime
                feasibility_current = feasibility_prime
            else:
                p = math.exp((cost_current - cost_prime) / T)
                rand1 = random.random()

                if rand1 < p:
                    S_current = (S_prime[0].copy(), cost_prime)
                    cost_current = cost_prime
                    feasibility_current = feasibility_prime

            iteration += 1

            # Check if the current solution is better than the best solution found so far
            if cost_prime < cost_best:
                S_best = (S_prime[0].copy(), cost_prime)  # update the best solution
                cost_best = cost_prime  # update the best cost
                feasibility_best = feasibility_prime  # update the best feasibility

                best_time = time.time() - start_time  # update the best time
                best_iter = iteration  # update the best iteration

                T_b = T  # update the best temperature
                plot_maze(S_best[0])

            # Update the gap between the current solution found so far and the benchmark cost
            gap = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
            if gap <= threshold_gap and threshold_gap_time is None and threshold_gap_iter is None:
                threshold_gap_time = time.time() - start_time
                threshold_gap_iter = iteration

        T = alpha * T  # decrease the temperature by the cooling rate

        # Check if the temperature is too low
        if T < threshold_temp:
            T_b = 2 * T_b
            T = min(T_b, T_max)
            Len = rollback_Len

        # add the current solution to the list of solutions
        solutions.append(S_current)
        solution_costs.append(cost_current)

        if print_iteration:
            # print the current iteration information""
            print("\n--------------------------------------------------")
            print("Iteration: ", iteration)
            print("Time elapsed: ", time.time() - start_time)
            print("Old solution: ", solutions[-2])
            print("Current solution: ", S_current)
            print("Current cost: ", cost_current)
            print("Feasibility: ", feasibility_current)
            print("Gap: ", gap)
            print("Temperature: ", T)
            print("Len: ", Len)
            print("\nBest solution: ", S_best)
            print("Best cost: ", cost_best)
            print("Best Feasibility: ", feasibility_best)
            print("Best gap: ", abs(cost_best - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf)
            print("Best time: ", best_time)
            print("Best iteration: ", best_iter)
            print("Best temperature: ", T_b)
            print("--------------------------------------------------\n")

    # ----------------------------------------------- SIMULATED ANNEALING ----------------------------------------------

    time_elapsed = time.time() - start_time  # time elapsed since the start of the algorithm

    if print_results:
        print("-------------------- RESULTS --------------------")
        print("Run parameters:")
        print("T_0:", T_0)

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

        print("\nOther results:")

        print("\n--------------------------------------------------")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("Feasibility: ", feasibility_best)
        print("--------------------------------------------------\n")

    if plot_results:
        plt.plot(solution_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Current cost vs Iteration")
        plt.axhline(y=cost_best, color='r', linestyle='-')
        plt.show()

    run_params = {"T_0": T_0,
                  "alpha": alpha}

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

    alg_specific_results = {'T_b': T_b,
                            'T_max': T_max,
                            'Len': Len}

    if save_results:
        '''Holder.save_results(algorithm='SA',
                            algorithm_params=run_params,
                            classic_results=classic_results,
                            algorithm_results=alg_specific_results,
                            path=save_results_path,
                            file_name=save_results_file_name)'''
        pass

    return S_best


def check_parameters(N_List: list[Callable],
                     init_temp: float = 0.5,
                     cooling_rate: float = 0.99,
                     N_func_weights: Union[str, list[float | int]] = 'uniform',
                     save_results: bool = False,
                     save_results_path: str = '',
                     save_results_file_name: str = '') -> None:
    # Number of neighbor functions
    no_functions: int = len(N_List)

    # check the parameters
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

    if init_temp < 0 or init_temp > 1:
        raise ValueError("The initial temperature must be between 0 and 1")

    if cooling_rate < 0 or cooling_rate > 1:
        raise ValueError("The cooling rate must be between 0 and 1")

    if save_results:
        if save_results_path == '':
            msg = "The path to save the results is empty."
            raise ValueError(msg)
        elif save_results_file_name == '':
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

    if cost_best < 0.5:
        return True

    return False
