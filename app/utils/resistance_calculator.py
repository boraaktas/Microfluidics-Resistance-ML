import cplex

from .constants import Constants
from .tile_type import TileType


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
        cur_tile_type_letter = get_tile_type_letter(flat_list[j][1])

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
            entry_point_pressure = other_elements[0][2]

            exit_point_num = int(other_elements_with_labels[-1][1:])
            exit_point_pressure = other_elements[-1][2]

            rl_count = 0
            rc_count = 0

            q_values = None

            for k in range(len(other_elements_with_labels)):
                element = other_elements_with_labels[k]
                if element[:2] == 'RL':
                    rl_count += 1
                elif element[:2] == 'RC':
                    rc_count += 1
                if element[:1] == 'Q' and other_elements[k][2] is not None:
                    q_values = other_elements[k][2]

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
            formatted_lines += f"{indent_str}[{line[0]}, TileType.{line[1].name}, {line[2]}],\n"
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


def calculate_resistance_with_cplex(circuit,
                                    add_objective=True):
    A, b, x, RLS_and_RCS_COUNT = create_matrix(circuit)

    # create the model
    model = cplex.Cplex()
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)

    # add variables
    for i in range(len(x)):
        x_i = x[i]
        # if x_i starts with P, it is a pressure, and there is no upper bound
        if x_i[0] == 'P':
            model.variables.add(obj=[0],
                                lb=[0],
                                names=[x_i])
        # if x_i starts with R, it is a resistance, then it should be greater than 1 and less than 60
        else:
            if x_i in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x_i] == 0:
                model.variables.add(obj=[0],
                                    lb=[0],
                                    ub=[0],
                                    names=[x_i])
            else:
                model.variables.add(obj=[0],
                                    lb=[1],
                                    ub=[70],
                                    names=[x_i])

    # add a variable to save the maximum resistance value
    max_resistance_var = 'max_resistance'
    model.variables.add(obj=[0], lb=[0], ub=[70], names=[max_resistance_var])

    # add a variable to save the minimum resistance value
    min_resistance_var = 'min_resistance'
    model.variables.add(obj=[0], lb=[0], ub=[70], names=[min_resistance_var])

    # add constraints
    for i in range(len(A)):
        model.linear_constraints.add(lin_expr=[[[x[j] for j in range(len(x))], A[i]]],
                                     senses=['E'],
                                     rhs=[b[i]])

    # add constraints to find the maximum and minimum resistance values
    for i in range(len(x)):
        if x[i][:2] == 'RC' or x[i][:2] == 'RL':
            if x[i] in RLS_and_RCS_COUNT and RLS_and_RCS_COUNT[x[i]] == 0:
                continue
            # max_resistance >= R[i]
            model.linear_constraints.add(lin_expr=[[[max_resistance_var, x[i]], [1, -1]]],
                                         senses=['G'],
                                         rhs=[0])
            # min_resistance <= R[i]
            model.linear_constraints.add(lin_expr=[[[min_resistance_var, x[i]], [1, -1]]],
                                         senses=['L'],
                                         rhs=[0])

    # set the objective
    # set objective as the difference between the maximum and minimum resistance values in the circuit to minimize it
    if add_objective:
        model.variables.add(obj=[1], names=['obj_diff'])
        model.linear_constraints.add(lin_expr=[[[max_resistance_var, min_resistance_var, 'obj_diff'], [1, -1, -1]]],
                                     senses=['E'],
                                     rhs=[0])
        model.objective.set_name('obj_diff')
    model.objective.set_sense(model.objective.sense.minimize)

    # solve the model
    model.solve()

    # get the results
    results = {}
    for i in range(len(x)):
        results[x[i]] = round(model.solution.get_values(x[i]), 3)

    return results


def make_dict_reverse(dictionary):
    unique_values = set(dictionary.values())
    new_dict = {value: [] for value in unique_values}
    for key in dictionary.keys():
        new_dict[dictionary[key]].append(key)

    return new_dict


def match_resistance_dict_and_label_dict(resistance_dict, label_dict):
    new_dict = {}
    for key in resistance_dict.keys():
        labels: list[str] = resistance_dict[key]
        new_dict[key] = []
        for label in labels:
            cells = label_dict[label]
            for cell in cells:
                new_dict[key].append((cell[0], cell[1]))

    updated_dict = {}
    for key in new_dict.keys():
        for cell in new_dict[key]:
            # if cell type is straight horizontal or vertical check key if (key, L) or (key, C) is in the dictionary
            if cell[1] == TileType.STRAIGHT_HORIZONTAL or cell[1] == TileType.STRAIGHT_VERTICAL:
                if (key, 'STRAIGHT') in updated_dict:  # add position of the cell to the list
                    updated_dict[(key, 'STRAIGHT')].append((cell[0], cell[1]))
                else:
                    updated_dict[(key, 'STRAIGHT')] = [(cell[0], cell[1])]

            elif cell[1] == TileType.TURN_WEST_SOUTH or cell[1] == TileType.TURN_WEST_NORTH or \
                    cell[1] == TileType.TURN_EAST_SOUTH or cell[1] == TileType.TURN_EAST_NORTH:
                if (key, 'CORNER') in updated_dict:  # add position of the cell to the list
                    updated_dict[(key, 'CORNER')].append((cell[0], cell[1]))
                else:
                    updated_dict[(key, 'CORNER')] = [(cell[0], cell[1])]

            else:
                raise ValueError("Invalid Tile Type")

    return updated_dict


def calculate_resistance(circuit):
    circuit_results = {}
    for single_circuit in circuit:
        mf_circuit_results_cplex_with_obj = calculate_resistance_with_cplex(circuit=single_circuit,
                                                                            add_objective=True)
        circuit_results.update(mf_circuit_results_cplex_with_obj)

    results_dict = {}
    for key in circuit_results.keys():
        if key.startswith('RL') or key.startswith('RC'):
            if circuit_results[key] != 0:
                results_dict[key] = circuit_results[key]

    reverse_dict = make_dict_reverse(results_dict)

    return reverse_dict


def R_calculator(nested_list):
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

    RESISTANCES = calculate_resistance(NEW_CIRCUIT)
    LABEL_DICT_REVERSED = make_dict_reverse(LABEL_DICT)

    RESULTS = match_resistance_dict_and_label_dict(RESISTANCES, LABEL_DICT_REVERSED)

    return RESULTS
