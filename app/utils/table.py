from .constants import Constants
from .tile import Tile
from .tile_type import TileType


class Table:

    def __init__(self, table_x: int, table_y: int):
        self.table_x = table_x
        self.table_y = table_y
        self.table = [[Tile() for _ in range(table_y)] for _ in range(table_x)]
        self.start_with_initial_table()

    def start_with_initial_table(self):

        data = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 16, 11, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 16, 21, 11, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 16, 11, 13, 15, 11, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 11, 11, 11, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 15, 11, 11, 11, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 11, 11, 11, 9, 11, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]

        """

        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    16   11   9    6    0    0    0    0    0    0    0    0    0    
        0    0    16   11   11   11   11   13   0    0    0    0    0    0    0    0    0    0    0    0    
        1    11   19   14   16   11   11   11   11   11   20   11   9    6    0    0    0    0    0    0    
        0    0    0    15   18   0    0    0    0    0    12   0    0    0    0    0    0    0    0    0    
        0    0    0    0    12   0    0    0    0    0    10   0    0    0    0    0    0    0    0    0    
        0    5    9    14   12   0    0    0    0    0    7    0    0    0    0    0    0    0    0    0    
        0    0    0    12   12   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    16   19   13   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    12   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    10   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    7    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    
        0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
        """

        """data = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 16, 11, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 16, 11, 11, 11, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 11, 19, 14, 16, 11, 11, 11, 11, 11, 20, 11, 9, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 15, 18, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 9, 14, 12, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 16, 19, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]"""

        for i in range(self.table_x):
            for j in range(self.table_y):
                self.set_tile(i, j, Tile(TileType(data[i][j])))

    def make_all_tiles_attributes_none(self):
        for i in range(self.table_x):
            for j in range(self.table_y):
                self.table[i][j].make_all_none()

    def set_tile(self, x: int, y: int, tile: Tile):

        if x < 0 or y < 0 or x >= self.table_x or y >= self.table_y:
            raise Exception("Index out of bounds")

        self.table[x][y].set_tile_type(tile.tile_type)

    def find_entry_pressures(self):
        table = self.table
        table_x = self.table_x
        table_y = self.table_y

        entry_pressures = []
        for i in range(table_x):
            for j in range(table_y):
                if table[i][j].tile_type == TileType.START_EAST or table[i][j].tile_type == TileType.START_NORTH or \
                        table[i][j].tile_type == TileType.START_SOUTH or table[i][j].tile_type == TileType.START_WEST:
                    entry_pressures.append(((i, j), table[i][j]))

        # sort the entry pressures according to first the y coordinate then the x coordinate
        entry_pressures.sort(key=lambda x: (x[0][0], x[0][1]))

        return entry_pressures

    def find_flow_rate_calculators(self):
        table = self.table
        table_x = self.table_x
        table_y = self.table_y

        flow_rate_calculators = []
        for i in range(table_x):
            for j in range(table_y):
                if table[i][j].tile_type == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL or \
                        table[i][j].tile_type == TileType.FLOW_RATE_CALCULATOR_VERTICAL:
                    flow_rate_calculators.append(((i, j), table[i][j]))

        # sort the flow rate calculators according to first the y coordinate then the x coordinate
        flow_rate_calculators.sort(key=lambda x: (x[0][0], x[0][1]))

        return flow_rate_calculators

    def find_exit_pressures(self):
        table = self.table
        table_x = self.table_x
        table_y = self.table_y

        exit_pressures = []
        for i in range(table_x):
            for j in range(table_y):
                if table[i][j].tile_type == TileType.EAST_END or table[i][j].tile_type == TileType.WEST_END or \
                        table[i][j].tile_type == TileType.NORTH_END or table[i][j].tile_type == TileType.SOUTH_END:
                    exit_pressures.append(((i, j), table[i][j]))

        # sort the exit pressures according to first the y coordinate then the x coordinate
        exit_pressures.sort(key=lambda x: (x[0][0], x[0][1]))

        return exit_pressures

    def find_divisions(self):
        table = self.table
        table_x = self.table_x
        table_y = self.table_y

        divisions = []
        for i in range(table_x):
            for j in range(table_y):
                if (table[i][j].tile_type == TileType.DIVISION_2_FROM_EAST
                        or table[i][j].tile_type == TileType.DIVISION_2_FROM_WEST
                        or table[i][j].tile_type == TileType.DIVISION_2_FROM_NORTH
                        or table[i][j].tile_type == TileType.DIVISION_2_FROM_SOUTH
                        or table[i][j].tile_type == TileType.DIVISION_3):
                    divisions.append(((i, j), table[i][j]))

        # sort the exit pressures according to first the y coordinate then the x coordinate
        divisions.sort(key=lambda x: (x[0][0], x[0][1]))

        return divisions

    def transform_table(self):

        table_x = self.table_x
        table_y = self.table_y

        self.make_all_tiles_attributes_none()

        table = self.table.copy()
        # print the table
        for i in range(table_x):
            for j in range(table_y):
                table_obj_i_j_value = table[i][j].tile_type.value
                end = " " * (5 - len(str(table_obj_i_j_value)))
                print(table_obj_i_j_value, end=end)
            print()

        # first find the starter points
        starter_points = []
        for i in range(table_x):
            for j in range(table_y):
                if table[i][j].tile_type in Constants.STARTER_TYPES:
                    starter_points.append(((i, j), table[i][j]))

        lines = []
        visited_tiles = []
        for starter_point in starter_points:
            starter_point_x, starter_point_y = starter_point[0]
            cur_line = []
            new_line = self.find_line(starter_point_x,
                                      starter_point_y,
                                      cur_line,
                                      "start",
                                      visited_tiles)
            lines.append(new_line)

        print("Lines:")
        print(f"[\n{self.format_lines(lines)}]")
        print("------------------------------------------------")
        print("Visited tiles:")
        print(visited_tiles)
        print("------------------------------------------------")

        return lines

    def format_lines(self, lines, indent=0):
        formatted_lines = ""
        indent_str = "  " * indent

        print("[")
        for line in lines:
            if isinstance(line[0], tuple):
                formatted_lines += f"{indent_str}[{line[0]}, {line[1]}],\n"
            else:
                formatted_lines += f"{indent_str}[\n{self.format_lines(line, indent + 1)}{indent_str}],\n"
        print("]")

        return formatted_lines

    def find_line(self,
                  cur_x: int,
                  cur_y: int,
                  cur_line,  #: list[list[tuple[int, int], TileType, Union[int]]],
                  coming_direction: str,
                  visited_tiles: list[tuple[int, int]]
                  ) -> list[tuple[tuple[int, int], TileType]]:

        current_tile = self.table[cur_x][cur_y]
        current_tile.coming_direction = coming_direction
        cur_line.append([(cur_x, cur_y), current_tile])
        visited_tiles.append((cur_x, cur_y))

        try:
            current_tile_going_directions = current_tile.find_going_direction(coming_direction)
            current_tile.going_directions = current_tile_going_directions
        except Exception as e:
            raise Exception(e)

        if current_tile.tile_type in Constants.DIVISION_TYPES:
            del cur_line[-1]
            cur_line.append([(cur_x, cur_y), current_tile])

        if current_tile_going_directions == ["end"]:
            # delete the last element of the line
            del cur_line[-1]
            cur_line.append([(cur_x, cur_y), current_tile])
            return cur_line

        elif current_tile_going_directions == ["empty"]:
            raise Exception("Empty tile found")

        else:
            branches = []
            for direction in current_tile_going_directions:
                next_tile_x, next_tile_y = Table.get_next_tile_coordinates(cur_x, cur_y, direction)
                next_tile = self.table[next_tile_x][next_tile_y]
                next_tile_coming_direction = Constants.OPPOSITE_DIRECTIONS[direction]

                if next_tile_coming_direction in Constants.ALLOWED_COMING_DIRECTIONS[next_tile.tile_type]:
                    if (next_tile_x, next_tile_y) in visited_tiles:
                        raise Exception("Loop found")

                    branch = self.find_line(next_tile_x, next_tile_y, [], next_tile_coming_direction, visited_tiles)
                    branches.append(branch)

            if len(branches) > 1:
                for branch in branches:
                    cur_line.append([[(cur_x, cur_y), current_tile]] + branch)
            else:
                cur_line.extend(branches[0])

        return cur_line

    def flat_nested_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if not isinstance(item[0], tuple):
                flat_list.extend(self.flat_nested_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def make_nested_list_to_list_of_lists(self, nested_list):

        flat_list = self.flat_nested_list(nested_list)

        l_of_l = [[]]
        for item in flat_list:
            item_obj = item[1]
            if item_obj.tile_type in Constants.STARTER_TYPES:
                l_of_l[-1].append(item)
            elif item_obj.tile_type in Constants.END_TYPES:  # add a new empty list
                l_of_l[-1].append(item)
                l_of_l.append([])
            elif item_obj.tile_type not in Constants.P_TILES:
                l_of_l[-1].append(item)
            elif item_obj.tile_type in Constants.DIVISION_TYPES and len(l_of_l[-1]) != 0:
                l_of_l[-1].append(item)
                l_of_l.append([])
            elif item_obj.tile_type in Constants.DIVISION_TYPES and len(l_of_l[-1]) == 0:
                l_of_l[-1].append(item)

        # remove last empty list
        l_of_l = l_of_l[:-1]

        return l_of_l

    def set_selected_comb_to_all_components(self, transformed_table):

        for circuit in transformed_table:

            flattened_circuit = self.flat_nested_list(circuit)

            selected_comb = None
            for i in range(len(flattened_circuit)):
                # if the tile has selected_comb_for_tile attribute
                if flattened_circuit[i][1].selected_comb_for_tile is not None:
                    selected_comb = flattened_circuit[i][1].selected_comb_for_tile
                    break

            # put all the selected_comb_for_tile to all the components
            for i in range(len(flattened_circuit)):
                coords_tile = flattened_circuit[i][0]
                self.table[coords_tile[0]][coords_tile[1]].selected_comb_for_tile = selected_comb

    def set_flow_rates_for_other_components(self, transformed_table):

        list_of_lists = self.make_nested_list_to_list_of_lists(transformed_table)

        for l_l in list_of_lists:

            found_flow_rate = None
            for i in range(len(l_l)):
                # pass the first element
                if i == 0:
                    continue
                if l_l[i][1].flow_rate_in_this_cell is not None:
                    found_flow_rate = l_l[i][1].flow_rate_in_this_cell
                    break

            # set the flow rate for all the components
            # if the first element of the list is a divider, then do not set the flow rate
            for i in range(len(l_l)):
                if i == 0 and l_l[i][1].tile_type in Constants.DIVISION_TYPES:
                    continue
                coords_tile = l_l[i][0]
                self.table[coords_tile[0]][coords_tile[1]].flow_rate_in_this_cell = found_flow_rate

    def find_resistance_between_two_points(self, transformed_table):
        serial_resistance = 0
        parallel_resistance = 0
        for i in range(len(transformed_table)):
            cur_element = transformed_table[i]
            # if the first element of the current element is a tuple
            if isinstance(cur_element[0], tuple):
                cur_element_resistance = cur_element[1].generated_resistance_in_this_cell
                if cur_element_resistance is not None:
                    serial_resistance += cur_element[1].generated_resistance_in_this_cell
            else:
                cur_element_resistance = self.find_resistance_between_two_points(cur_element)
                parallel_resistance += 1 / cur_element_resistance

        total_res = serial_resistance + 1 / parallel_resistance if parallel_resistance != 0 else serial_resistance

        return total_res

    def set_generated_flow_rates_between_two_points(self, transformed_table, current_flow_rate):

        parallel_resistances = []
        for i in range(len(transformed_table)):
            cur_element = transformed_table[i]
            if isinstance(cur_element[0], tuple):
                if not cur_element[1].tile_type in Constants.DIVISION_TYPES:
                    cur_element[1].generated_flow_rate_in_this_cell = current_flow_rate
            else:
                parallel_resistances.append(self.find_resistance_between_two_points(cur_element))

        total_res_in_parallels = sum(parallel_resistances)

        current_parallel_branch = 0
        for i in range(len(transformed_table)):
            cur_element = transformed_table[i]
            if not isinstance(cur_element[0], tuple):
                going_flow_rate = current_flow_rate * ((total_res_in_parallels - parallel_resistances[
                    current_parallel_branch]) / total_res_in_parallels)
                self.set_generated_flow_rates_between_two_points(cur_element, going_flow_rate)
                current_parallel_branch += 1

    @staticmethod
    def find_entry_flow_rate(transformed_table, total_resistance):

        entry_pressure = transformed_table[0][1].pressure_in_this_cell

        entry_flow_rate = entry_pressure / total_resistance

        return entry_flow_rate

    def set_generated_flow_rates(self, transformed_table):
        for transformed_table_circuit in transformed_table:
            total_resistance_in_current_circuit = self.find_resistance_between_two_points(transformed_table_circuit)
            entry_flow_rate_in_current_circuit = self.find_entry_flow_rate(transformed_table_circuit,
                                                                           total_resistance_in_current_circuit)
            self.set_generated_flow_rates_between_two_points(transformed_table_circuit,
                                                             entry_flow_rate_in_current_circuit)
            print(f"Total resistance in current circuit: {total_resistance_in_current_circuit}")
            print(f"Entry flow rate in current circuit: {entry_flow_rate_in_current_circuit}")
            print()

    @staticmethod
    def get_next_tile_coordinates(cur_x: int, cur_y: int, direction: str) -> tuple[int, int]:
        if direction == "north":
            return cur_x - 1, cur_y
        elif direction == "south":
            return cur_x + 1, cur_y
        elif direction == "west":
            return cur_x, cur_y - 1
        elif direction == "east":
            return cur_x, cur_y + 1
        else:
            raise Exception(f"Direction: {direction} is not allowed")
