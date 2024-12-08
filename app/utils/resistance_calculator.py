import pickle

import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Constraint, RangeSet, minimize, value

from .constants import Constants
from .tile_type import TileType

from .helper_functions import resource_path


def make_flat_list(circuit):
    flat_list = []
    for i in range(len(circuit)):
        if isinstance(circuit[i][0], tuple):
            flat_list.append(tuple(circuit[i]))
        elif isinstance(circuit[i], list):
            flat_list.extend(make_flat_list(circuit[i]))
        else:
            raise ValueError("Invalid Circuit")
    return flat_list


def get_tile_type_letter(tile_type):
    if tile_type in Constants.RL_TILES:
        return "RL"
    elif tile_type in Constants.RC_TILES:
        return "RC"
    elif tile_type in Constants.Q_TILES:
        return "Q"
    elif tile_type in Constants.P_TILES:
        return "P"
    else:
        return None


def label_nested_list(circuit, label_dict, i):
    # make nested list a flat list
    flat_list = make_flat_list(circuit)

    for j in range(len(flat_list)):
        print(flat_list[j])
        cur_tile = flat_list[j]
        cur_tile_type_letter = get_tile_type_letter(flat_list[j][1].tile_type)

        if j == 0:
            label_dict[cur_tile] = cur_tile_type_letter + str(i)
            i += 1
        else:
            if cur_tile_type_letter == "RL":
                label_dict[cur_tile] = cur_tile_type_letter + str(i)
            elif cur_tile_type_letter == "RC":
                label_dict[cur_tile] = cur_tile_type_letter + str(i)
            elif cur_tile_type_letter == "Q":
                label_dict[cur_tile] = cur_tile_type_letter + str(i)
            elif cur_tile_type_letter == "P":
                # check if cur_tile is in the result dictionary
                if cur_tile in label_dict:
                    i += 1
                else:
                    label_dict[cur_tile] = cur_tile_type_letter + str(i)

    print()
    for k, v in label_dict.items():
        print(f"{k}: {v}")
    return label_dict, i


def rewrite_circuit_with_labels(circuit, label_dict):
    new_circuit = []
    next_i = 0
    for i in range(len(circuit)):
        if i < next_i:
            continue
        cur_element = circuit[i]

        if isinstance(circuit[i][0], tuple):

            other_elements = []  # get the other elements until the next list or the end of the list
            j = i
            while j < len(circuit) and not isinstance(circuit[j][0], list):
                other_elements.append(circuit[j])
                j += 1
            # update i
            next_i = j

            # update the new circuit
            other_elements_with_labels = []
            for other_element in other_elements:
                other_elements_with_labels.append(label_dict[tuple(other_element)])

            entry_point_num = int(other_elements_with_labels[0][1:])
            entry_point_pressure = other_elements[0][1].pressure_in_this_cell

            exit_point_num = int(other_elements_with_labels[-1][1:])
            exit_point_pressure = other_elements[-1][1].pressure_in_this_cell

            rl_count = 0
            rc_count = 0

            q_values = None

            for k in range(len(other_elements_with_labels)):
                element = other_elements_with_labels[k]
                if element[:2] == 'RL':
                    rl_count += 1
                elif element[:2] == 'RC':
                    rc_count += 1

                if ((element[:1] == 'Q' or element[:1] == 'P')
                        and other_elements[k][1].flow_rate_in_this_cell is not None):
                    q_values = other_elements[k][1].flow_rate_in_this_cell

            shrinked_elements = (('P' + str(entry_point_num), entry_point_pressure),
                                 ('RL' + str(exit_point_num), rl_count, None),
                                 ('RC' + str(exit_point_num), rc_count, None),
                                 ('Q' + str(exit_point_num), q_values),
                                 ('P' + str(exit_point_num), exit_point_pressure))

            new_circuit.append(shrinked_elements)

        elif isinstance(circuit[i], list):
            new_circuit.append(rewrite_circuit_with_labels(cur_element, label_dict))

        else:
            raise ValueError("Invalid Circuit")

    return new_circuit


def format_lines(lines, indent=0):
    formatted_lines = ""
    indent_str = "  " * indent

    for line in lines:
        if isinstance(line[0], tuple):
            formatted_lines += f"{indent_str}[{line[0]}, {line[1]}],\n"
        else:
            formatted_lines += f"{indent_str}[\n{format_lines(line, indent + 1)}{indent_str}],\n"

    return formatted_lines


def format_lines_2(lines, indent=0):
    formatted_lines = ""
    indent_str = "  " * indent

    for line in lines:
        if isinstance(line, list):
            formatted_lines += f"{indent_str}[\n{format_lines_2(line, indent + 1)}{indent_str}],\n"
        else:
            formatted_lines += f"{indent_str}{line},\n"

    return formatted_lines


def get_all_x(all_lines):
    variables = []
    for i in range(len(all_lines)):
        for j in range(len(all_lines[i])):
            if j == len(all_lines[i]) - 1:
                line_no = all_lines[i][j][0][1:]
                variables.append(f"R{line_no}")
            if all_lines[i][j][0] not in variables and (all_lines[i][j][0][0] == 'P' or all_lines[i][j][0][0] == 'R'):
                variables.append(all_lines[i][j][0])

    return variables


def get_all_lines(circuit):
    all_lines = []
    for i in range(len(circuit)):
        if isinstance(circuit[i], tuple):
            all_lines.append(circuit[i])
        elif isinstance(circuit[i], list):
            all_lines += get_all_lines(circuit[i])
        else:
            raise ValueError("Invalid Circuit")

    return all_lines


def create_matrix(circuit):
    all_lines = get_all_lines(circuit)
    for i in range(len(all_lines)):
        print(all_lines[i])

    x = get_all_x(all_lines)
    print(x)

    A = []
    b = []

    RLS_and_RCS_COUNT = {}

    for i in range(len(all_lines)):
        line = all_lines[i]

        entry_pressure = line[0][0]
        entry_pressure_val = line[0][1]
        entry_pressure_index = x.index(entry_pressure)

        exit_pressure = line[-1][0]
        exit_pressure_val = line[-1][1]
        exit_pressure_index = x.index(exit_pressure)

        RL = line[1][0]
        RL_count = line[1][1]
        RL_index = x.index(RL)
        RLS_and_RCS_COUNT[RL] = RL_count

        RC = line[2][0]
        RC_count = line[2][1]
        RC_index = x.index(RC)
        RLS_and_RCS_COUNT[RC] = RC_count

        R = f"R{exit_pressure[1:]}"
        R_index = x.index(R)

        # if entry pressure val is not None
        if entry_pressure_val is not None:
            A.append([1 if x[j] == entry_pressure else 0 for j in range(len(x))])
            b.append(entry_pressure_val)
            print(f"{x[entry_pressure_index]}: {entry_pressure_val}")

        # if exit pressure val is not None
        if exit_pressure_val is not None:
            A.append([1 if x[j] == exit_pressure else 0 for j in range(len(x))])
            b.append(exit_pressure_val)
            print(f"{x[exit_pressure_index]}: {exit_pressure_val}")

        # if RL_count is 0, then RL should be 0
        if RL_count == 0:
            A.append([1 if x[j] == RL else 0 for j in range(len(x))])
            b.append(0)
            print(f"{x[RL_index]}: 0")

        # if RC_count is 0, then RC should be 0
        if RC_count == 0:
            A.append([1 if x[j] == RC else 0 for j in range(len(x))])
            b.append(0)
            print(f"{x[RC_index]}: 0")

        flow_rate_val = line[3][1]

        # add the equation (entry pressure - exit pressure - R * flow_rate = 0)
        new_row = [0] * len(x)
        new_row[entry_pressure_index] = 1
        new_row[exit_pressure_index] = -1
        new_row[R_index] = -flow_rate_val
        A.append(new_row)
        b.append(0)
        print(f"{x[entry_pressure_index]} - {x[exit_pressure_index]} - ({flow_rate_val}*{x[R_index]}) = 0")

        # R = RL1 * RL_count + RC1 * RC_count
        new_row = [0] * len(x)
        new_row[R_index] = 1
        new_row[RL_index] = -RL_count
        new_row[RC_index] = -RC_count
        A.append(new_row)
        b.append(0)
        print(f"{x[R_index]} - ({RL_count}*{x[RL_index]}) - ({RC_count}*{x[RC_index]}) = 0")

    print()
    # print variable names
    for x_i in x:
        len_x_i = len(x_i)
        end = " " * (6 - len_x_i)
        print(x_i, end=end)
    print(" | b")

    print("-" * (6 * len(x) + 3))

    for i in range(len(A)):
        for j in range(len(A[i])):
            A_ij = A[i][j]
            len_a_ij = len(str(A_ij))
            end = " " * (6 - len_a_ij)
            print(A_ij, end=end)
        print(f" | {b[i]}")

    print()

    return A, b, x, RLS_and_RCS_COUNT


def flatten_circuit(circuit):
    new_circuit = []
    for i in range(len(circuit)):
        if isinstance(circuit[i], tuple):
            new_circuit.append(circuit[i])
        elif isinstance(circuit[i], list):
            new_circuit.extend(flatten_circuit(circuit[i]))
        else:
            raise ValueError("Invalid Circuit")
    return new_circuit


def calculate_resistance_via_optimization(circuit,
                                          resistance_bounds,
                                          obj_type="farthest_to_lb",
                                          start_num=None,
                                          end_num=None) -> dict[tuple[str, int], float]:

    A, b, x, RLS_and_RCS_COUNT = create_matrix(circuit)

    flatten_cir = flatten_circuit(circuit)
    print("Flatten Circuit:")
    print(flatten_cir)
    R_div_counts = {}
    for var in x:
        if var[0] == 'R' and var not in RLS_and_RCS_COUNT:
            var_num = int(var[1:])
            count = 0
            for line in flatten_cir:
                if var_num == int(line[-1][0][1:]):
                    if line[-1][1] is None:
                        count += 1
                    if line[0][1] is None:
                        count += 1
            R_div_counts[var] = count

    R_indices = [x.index(var) for var in R_div_counts]

    if obj_type not in ["diff_min_max", "farthest_to_avg", "farthest_to_lb", "farthest_to_ub"]:
        raise ValueError("obj_type should be one of the following: diff_min_max, farthest_to_avg, "
                         "farthest_to_lb, farthest_to_ub")

    farthest_type = None
    if obj_type.startswith("farthest"):
        farthest_type = obj_type.split("_")[-1]

    if start_num is None:
        start_num = 0
    if end_num is None:
        end_num = len(resistance_bounds)

    resistance_bounds = dict(list(resistance_bounds.items())[start_num:end_num])

    all_upper_bounds = [resistance_bounds[i]['ub'] for i in resistance_bounds]
    all_lower_bounds = [resistance_bounds[i]['lb'] for i in resistance_bounds]

    # Create a Pyomo model
    model = ConcreteModel("ResistanceCalculator")

    # Define sets
    model.i = RangeSet(0, len(x) - 1)  # index for x
    model.j = RangeSet(min(resistance_bounds.keys()), max(resistance_bounds.keys()))  # index for combination

    # Define variables
    for i in model.i:
        x_i = x[i]
        if x_i[0] == 'P':
            for j in model.j:
                model.add_component(f"{x_i}_{j}", Var(domain=pyo.NonNegativeReals))
        else:
            for j in model.j:
                if x_i in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x_i] == 0:
                    model.add_component(f"{x_i}_{j}", Var(domain=pyo.NonNegativeReals,
                                                          bounds=(0, 0)))
                elif x_i in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x_i] > 0:
                    model.add_component(f"{x_i}_{j}", Var(domain=pyo.NonNegativeReals,
                                                          bounds=(0, resistance_bounds[j]['ub'])))
                else:
                    model.add_component(f"{x_i}_{j}", Var(domain=pyo.NonNegativeReals))

    # Define combination variables as binary
    for j in model.j:
        model.add_component(f"combination_{j}", Var(domain=pyo.Binary))

        # add max_resistance and min_resistance variables for each combination
        if obj_type == "diff_min_max":
            model.add_component(f"max_resistance_{j}",
                                Var(domain=pyo.NonNegativeReals))
            model.add_component(f"min_resistance_{j}",
                                Var(domain=pyo.NonNegativeReals))
        elif obj_type.startswith("farthest"):
            model.add_component(f"farthest_{j}",
                                Var(domain=pyo.NonNegativeReals))

    if obj_type == "diff_min_max":
        model.add_component("max_resistance",
                            Var(domain=pyo.NonNegativeReals, bounds=(min(all_lower_bounds), max(all_upper_bounds))))
        model.add_component("min_resistance",
                            Var(domain=pyo.NonNegativeReals, bounds=(min(all_lower_bounds), max(all_upper_bounds))))
    elif obj_type.startswith("farthest"):
        model.add_component("farthest", Var(domain=pyo.NonNegativeReals, bounds=(0, max(all_upper_bounds))))

    # Add constraints
    model.combination_constraint = Constraint(expr=sum(getattr(model, f"combination_{m_j}") for m_j in model.j) == 1)

    # add upper and lower bounds for each resistance
    for i in model.i:
        x_i = x[i]
        if x_i in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x_i] > 0:
            for j in model.j:
                model.add_component(f"upper_bound_{x_i}_{j}",
                                    Constraint(expr=getattr(
                                        model, f"{x_i}_{j}") <= resistance_bounds[j]['ub'] * getattr(
                                        model, f"combination_{j}")))
                model.add_component(f"lower_bound_{x_i}_{j}",
                                    Constraint(expr=getattr(
                                        model, f"{x_i}_{j}") >= resistance_bounds[j]['lb'] * getattr(
                                        model, f"combination_{j}")))

    # Add A matrix constraints for each combination
    for j in model.j:
        for i in range(len(A)):

            A_row = A[i]
            right_hand_side = b[i]
            found = False
            for r_index in R_indices:
                if not found:
                    if A_row[r_index] != 0 and A_row[r_index] != 1:
                        right_hand_side = -2 * resistance_bounds[j]['div_res'] * A_row[r_index]
                        found = True

            model.add_component(f"A_constraint_{i}_{j}",
                                Constraint(expr=sum(A_row[k] * getattr(
                                    model, f"{x[k]}_{j}") for k in model.i) == right_hand_side * getattr(
                                    model, f"combination_{j}")))

    # Set max_resistance and min_resistance
    for j in model.j:
        # max resistance should be greater than or equal to all resistances in the combination
        for i in model.i:
            if x[i] in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x[i]] > 0:
                if obj_type == "diff_min_max":
                    model.add_component(f"max_resistance_constraint_{i}_{j}",
                                        Constraint(expr=getattr(model, f"max_resistance_{j}") >= getattr(
                                            model, f"{x[i]}_{j}")))
                    model.add_component(f"min_resistance_constraint_{i}_{j}",
                                        Constraint(expr=getattr(model, f"min_resistance_{j}") <= getattr(
                                            model, f"{x[i]}_{j}")))
                elif obj_type.startswith("farthest"):
                    # farthest_j is the farthest resistance in the combination from the average of this combination
                    # the distance between a point and the average is calculated by the absolute difference
                    model.add_component(f"farthest_constraint_{i}_{j}_1",
                                        Constraint(expr=getattr(model, f"{x[i]}_{j}") - getattr(
                                            model, f"farthest_{j}") <= resistance_bounds[j][farthest_type] * getattr(
                                            model, f"combination_{j}")))
                    """model.add_component(f"farthest_constraint_{i}_{j}_2",
                                        Constraint(expr=getattr(model, f"{x[i]}_{j}") + getattr(
                                            model, f"farthest_{j}") >= resistance_bounds[j][farthest_type] * getattr(
                                            model, f"combination_{j}")))"""

        # max resistance should be maximum of all max_resistance variables
        if obj_type == "diff_min_max":
            model.add_component(f"max_resistance_constraint_{j}",
                                Constraint(
                                    expr=getattr(model, f"max_resistance_{j}") <= getattr(model, f"max_resistance")))
            model.add_component(f"min_resistance_constraint_{j}",
                                Constraint(
                                    expr=getattr(model, f"min_resistance_{j}") <= getattr(model, f"min_resistance")))

            model.add_component(f"min_max_resistance_constraint_{j}",
                                Constraint(expr=getattr(model, f"min_resistance_{j}") <= getattr(
                                    model, f"max_resistance_{j}")))

            model.add_component(f"min_resistance_constraint_minimum_{j}",
                                Constraint(
                                    expr=getattr(model, f"min_resistance_{j}") >= resistance_bounds[j]['lb'] * getattr(
                                        model, f"combination_{j}")))
            model.add_component(f"max_resistance_constraint_maximum_{j}",
                                Constraint(
                                    expr=getattr(model, f"max_resistance_{j}") <= resistance_bounds[j]['ub'] * getattr(
                                        model, f"combination_{j}")))

        elif obj_type.startswith("farthest"):
            model.add_component(f"farthest_constraint_{j}",
                                Constraint(expr=getattr(model, f"farthest_{j}") <= getattr(model, f"farthest")))

    if obj_type == "diff_min_max":
        model.add_component("min_max_resistance_constraint",
                            Constraint(expr=model.min_resistance <= model.max_resistance))

    # Define the objective function
    if obj_type == "diff_min_max":
        model.obj = Objective(expr=model.max_resistance + model.min_resistance, sense=minimize)
    elif obj_type.startswith("farthest"):
        model.obj = Objective(expr=model.farthest, sense=minimize)

    # Solve the model
    solver_path = resource_path('data//solver//glpsol.exe')
    solver = SolverFactory('glpk', executable=solver_path)

    # write the model to a file by using variables with the same name
    # model.write(filename="resistance_calculator.lp", io_options = {"symbolic_solver_labels":True})

    result = solver.solve(model, tee=True)

    # Check the solver status
    if result.solver.status != pyo.SolverStatus.ok:
        raise ValueError("Check the solver status")
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError("Check the solver termination condition")

    # Get the results rounded to 4 decimal places
    results = {}
    for i in model.i:
        for j in model.j:
            results[f"{x[i]}_{j}"] = round(value(getattr(model, f"{x[i]}_{j}")), 4)

    for j in model.j:
        results[f"combination_{j}"] = round(value(getattr(model, f"combination_{j}")), 4)

        if obj_type == "diff_min_max":
            results[f"max_resistance_{j}"] = round(value(getattr(model, f"max_resistance_{j}")), 4)
            results[f"min_resistance_{j}"] = round(value(getattr(model, f"min_resistance_{j}")), 4)
        elif obj_type.startswith("farthest"):
            results[f"farthest_{j}"] = round(value(getattr(model, f"farthest_{j}")), 4)

    if obj_type == "diff_min_max":
        results["max_resistance"] = round(value(model.max_resistance), 4)
        results["min_resistance"] = round(value(model.min_resistance), 4)
    elif obj_type.startswith("farthest"):
        results["farthest"] = round(value(model.farthest), 4)

    new_dict = {}

    # all combinations
    all_combination_nums = [int(i.split("_")[-1]) for i in results if i.startswith("combination_")]

    for i in all_combination_nums:
        # find variables for the combination
        variables = [j for j in results if j.endswith(f"_{i}") and j.startswith("P")]
        variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("RL")])
        variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("RC")])
        variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("R") and j not in variables])

        if obj_type == "diff_min_max":
            variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("max_resistance")])
            variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("min_resistance")])
        elif obj_type.startswith("farthest"):
            variables.extend([j for j in results if j.endswith(f"_{i}") and j.startswith("farthest")])

        new_dict[i] = variables

    selected_combination = [i for i in all_combination_nums if results[f"combination_{i}"] == 1][0]
    final_results = {}
    for var in new_dict[selected_combination]:
        if var.startswith("R") or var.startswith("P"):
            final_results[(var.split("_")[0], selected_combination)] = results[var]

    return final_results


def make_dict_reverse(dictionary):
    unique_values = set(dictionary.values())
    new_dict = {value_new: [] for value_new in unique_values}
    for key in dictionary.keys():
        new_dict[dictionary[key]].append(key)

    return new_dict


def match_resistance_dict_and_label_dict(resistance_dict, label_dict):

    new_dict = {}
    for key in resistance_dict.keys():
        labels: list[tuple[str, int]] = resistance_dict[key]
        selected_comb_num = labels[0][1]
        new_dict[key, selected_comb_num] = []
        for label in labels:
            cells = label_dict[label[0]]
            for cell in cells:
                cell[1].selected_comb_for_tile = selected_comb_num
                new_dict[key, selected_comb_num].append((cell[0], cell[1]))

    updated_dict = {}
    for key in new_dict.keys():
        for cell in new_dict[key]:
            # if cell type is straight horizontal or vertical check key if (key, L) or (key, C) is in the dictionary
            if (cell[1].tile_type == TileType.STRAIGHT_HORIZONTAL
                    or cell[1].tile_type == TileType.STRAIGHT_VERTICAL):
                if (key[0], key[1], 'STRAIGHT') in updated_dict:  # add position of the cell to the list
                    updated_dict[(key[0], key[1], 'STRAIGHT')].append((cell[0], cell[1]))
                else:
                    updated_dict[(key[0], key[1], 'STRAIGHT')] = [(cell[0], cell[1])]

            elif cell[1].tile_type == TileType.TURN_WEST_SOUTH or cell[1].tile_type == TileType.TURN_WEST_NORTH or \
                    cell[1].tile_type == TileType.TURN_EAST_SOUTH or cell[1].tile_type == TileType.TURN_EAST_NORTH:
                if (key[0], key[1], 'CORNER') in updated_dict:  # add position of the cell to the list
                    updated_dict[(key[0], key[1], 'CORNER')].append((cell[0], cell[1]))
                else:
                    updated_dict[(key[0], key[1], 'CORNER')] = [(cell[0], cell[1])]

            else:
                raise ValueError("Invalid Tile Type")

    return updated_dict


def calculate_resistance(circuit, resistance_bounds: dict):
    circuit_results = {}
    for single_circuit in circuit:
        mf_circuit_results_opt_with_obj = calculate_resistance_via_optimization(
            circuit=single_circuit, resistance_bounds=resistance_bounds)
        circuit_results.update(mf_circuit_results_opt_with_obj)

    results_dict = {}
    for key in circuit_results.keys():
        if key[0].startswith('RL') or key[0].startswith('RC'):
            if circuit_results[key] != 0:
                results_dict[key] = circuit_results[key]

    reverse_dict = make_dict_reverse(results_dict)

    return reverse_dict


def R_calculator(nested_list, resistance_bounds: dict):
    print("Circuit:")
    print(f"[\n{format_lines(nested_list)}]")

    LABEL_DICT = {}
    INDEX = 0
    NEW_CIRCUIT = []
    for CIRCUIT in nested_list:
        LABEL_DICT, INDEX = label_nested_list(CIRCUIT, LABEL_DICT, INDEX)
        NEW_CIRCUIT_PIECE = rewrite_circuit_with_labels(CIRCUIT, LABEL_DICT)
        NEW_CIRCUIT.append(NEW_CIRCUIT_PIECE)
        INDEX += 1

    print("Dictionary:")
    for K, V in LABEL_DICT.items():
        print(f"{K}: {V}")

    print(NEW_CIRCUIT)
    print("\nNew Circuit:")
    print(format_lines_2(NEW_CIRCUIT))

    RESISTANCES = calculate_resistance(NEW_CIRCUIT, resistance_bounds)
    LABEL_DICT_REVERSED = make_dict_reverse(LABEL_DICT)

    RESULTS = match_resistance_dict_and_label_dict(RESISTANCES, LABEL_DICT_REVERSED)

    return RESULTS
