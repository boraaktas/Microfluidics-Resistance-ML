import copy
import re

import cplex
import numpy as np

from .tile_type import TileType


def add_labels(flow_rate_circuit, label, new_circuit):
    last_label_num = int(label.split('-')[-1])
    for i in range(len(flow_rate_circuit)):
        if isinstance(flow_rate_circuit[i][0], tuple):
            new_circuit.append((label, flow_rate_circuit[i][0], flow_rate_circuit[i][2]))
        elif isinstance(flow_rate_circuit[i], list):
            add_labels(flow_rate_circuit[i], label + str('-') + str(last_label_num), new_circuit)
            last_label_num += 1
        else:
            raise ValueError("Invalid Circuit")

    return new_circuit


def delete_other_elements(circuit):
    """
    Deletes other elements in the circuit other than flow rate calculators
    """
    new_circuit = []

    for i in range(len(circuit)):

        if isinstance(circuit[i][0], tuple):
            if circuit[i][1] == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL or \
                    circuit[i][1] == TileType.FLOW_RATE_CALCULATOR_VERTICAL or \
                    circuit[i][1] == TileType.FLOW_RATE_CALCULATOR:
                new_circuit.append(circuit[i])
        elif isinstance(circuit[i], list):
            new_circuit.append(delete_other_elements(circuit[i]))
        else:
            raise ValueError("Invalid Circuit")

    return new_circuit


def check_flow_rates(circuit):
    # delete other elements in the circuit
    flow_rate_circuit = delete_other_elements(copy.deepcopy(circuit))

    flow_rates_with_labels = add_labels(flow_rate_circuit, 'Q-1', [])
    # check if the flow rates are correct in terms of conservation of mass
    # first if a

    # first if there are flw rates with same label then they should be equal
    for i in range(len(flow_rates_with_labels)):
        for j in range(i + 1, len(flow_rates_with_labels)):
            if flow_rates_with_labels[i][0] == flow_rates_with_labels[j][0]:
                if flow_rates_with_labels[i][2] is not None and flow_rates_with_labels[j][2] is not None:
                    if flow_rates_with_labels[i][2] != flow_rates_with_labels[j][2]:
                        raise ValueError(f"Flow rates in the same line should be equal. "
                                         f"Flow Rate 1: {flow_rates_with_labels[i][2]}, "
                                         f"Flow Rate 2: {flow_rates_with_labels[j][2]}")

    # drop the elements with the same label
    new_flow_rates_with_labels = []
    for i in range(len(flow_rates_with_labels)):
        if i == 0:
            new_flow_rates_with_labels.append(flow_rates_with_labels[i])
        else:
            if flow_rates_with_labels[i][0] != flow_rates_with_labels[i - 1][0]:
                new_flow_rates_with_labels.append(flow_rates_with_labels[i])

    # make new_flow_rates_with_labels a dictionary {label: ((x, y), flow_rate)}
    flow_rate_dict = {}
    for i in range(len(new_flow_rates_with_labels)):
        flow_rate_dict[new_flow_rates_with_labels[i][0]] = [new_flow_rates_with_labels[i][1],
                                                            new_flow_rates_with_labels[i][2]]

    conserved_flow_rates = conservation_of_mass(flow_rate_dict)

    return conserved_flow_rates


def solve_flow_rate_matrix_with_cplex(variables, A, b):
    # create the cplex object
    prob = cplex.Cplex()
    prob.set_results_stream(None)
    prob.set_warning_stream(None)
    prob.set_error_stream(None)

    # add the variables
    for i in range(len(variables)):
        prob.variables.add(names=[variables[i]], obj=[0], lb=[0], ub=[cplex.infinity])

    # add the constraints
    for i in range(len(A)):
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=variables, val=A[i])], senses=['E'], rhs=[b[i]])

    # solve the problem
    prob.solve()

    return prob.solution.get_values()


def conservation_of_mass(flow_rate_dict):
    # check if the flow rates are correct in terms of conservation of mass
    # example dict:
    # Q-1: ((-1, -1), None)
    # Q-1-1: ((-1, -1), None)
    # Q-1-1-1: ((1, 9), 1)
    # Q-1-1-2: ((3, 9), 1)
    # Q-1-1-3: ((2, 9), 1)
    # Q-1-2: ((5, 8), 1)

    variables = flow_rate_dict.keys()

    A = []
    b = []

    # first add equalities such as Q-1-1-1 = 1
    for key in variables:
        if flow_rate_dict[key][1] is not None:
            A.append([0] * len(variables))
            A[-1][list(variables).index(key)] = 1
            b.append(flow_rate_dict[key][1])

    # then add conservation of mass equations
    for key in variables:
        # find other keys that are starting with the same key and contains -number extra
        # if key is Q-1 then other_keys should be Q-1-1, Q-1-2, Q-1-3 not Q-1-1-1, Q-1-1-2
        other_keys = []
        for other_key in variables:
            if re.match(f"{key}-[0-9]+$", other_key):
                other_keys.append(other_key)

        if len(other_keys) == 0:
            continue

        key_index = list(variables).index(key)  # it should be 1
        new_row = [0] * len(variables)
        new_row[key_index] = 1

        for other_key in other_keys:
            other_key_index = list(variables).index(other_key)
            new_row[other_key_index] = -1

        A.append(new_row)
        b.append(0)

    # solve the equations with numpy if there is just one solution returns True
    # if there are infinite solutions returns False
    # if there is no solution returns False
    try:

        # print the matrix
        print()
        # print variable names
        for x_i in variables:
            len_x_i = len(x_i)
            end = " " * (15 - len_x_i)
            print(x_i, end=end)
        print(" | b")

        print("-" * (6 * len(variables) + 3))

        for i in range(len(A)):
            for j in range(len(A[i])):
                A_ij = A[i][j]
                len_a_ij = len(str(A_ij))
                end = " " * (15 - len_a_ij)
                print(A_ij, end=end)
            print(f" | {b[i]}")

        print()

        variables_list = list(variables)
        x = solve_flow_rate_matrix_with_cplex(variables_list, A, b)

        solution_dict = {}
        for i in range(len(variables_list)):
            if x[i] == 0:
                msg = f"There should be more flow rates to conserve mass"
                print(msg)
                raise ValueError(msg)
            solution_dict[flow_rate_dict[variables_list[i]][0]] = x[i]

        print("Solution:")
        for key in solution_dict:
            print(f"{key}: {solution_dict[key]}")

        return solution_dict

    except np.linalg.LinAlgError:
        print("No Solution")
        raise ValueError("No Solution")


def Q_calculator(transformed_table):
    copy_transformed_table = copy.deepcopy(transformed_table)
    CALCULATED_FLOW_RATES = {}
    for CIRCUIT in copy_transformed_table:
        calculated_flow_rate = check_flow_rates(CIRCUIT)
        CALCULATED_FLOW_RATES.update(calculated_flow_rate)

    print("CALCULATED FLOW RATES:")
    for k, v in CALCULATED_FLOW_RATES.items():
        print(f"{k}: {v}")

    return CALCULATED_FLOW_RATES
