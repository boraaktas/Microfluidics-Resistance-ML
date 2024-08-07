import copy

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

        for i in range(self.table_x):
            for j in range(self.table_y):
                self.set_tile(i, j, Tile(TileType(data[i][j])))

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

    def transform_table(self):

        table_x = self.table_x
        table_y = self.table_y

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
                formatted_lines += f"{indent_str}[{line[0]}, TileType.{line[1].name}, {line[2]}],\n"
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
        cur_line.append([(cur_x, cur_y), current_tile.tile_type, None])
        visited_tiles.append((cur_x, cur_y))

        try:
            current_tile_going_directions = current_tile.find_going_direction(coming_direction)
        except Exception as e:
            raise Exception(e)

        if current_tile.tile_type in Constants.DIVISION_TYPES:
            del cur_line[-1]
            cur_line.append([(cur_x, cur_y), TileType.FLOW_RATE_CALCULATOR, None])
            cur_line.append([(cur_x, cur_y), current_tile.tile_type, None])

        if current_tile_going_directions == ["end"]:
            # delete the last element of the line
            del cur_line[-1]
            cur_line.append([(cur_x, cur_y), TileType.FLOW_RATE_CALCULATOR, None])
            cur_line.append([(cur_x, cur_y), current_tile.tile_type, None])
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
                    cur_line.append([[(cur_x, cur_y), current_tile.tile_type, None]] + branch)
            else:
                cur_line.extend(branches[0])

        return cur_line

    def update_transformed_table_helper(self, transformed_table, x, y, value, mode):

        if mode == 1:
            if isinstance(transformed_table[0], tuple):
                if transformed_table[0] == (x, y):
                    if transformed_table[1] != TileType.FLOW_RATE_CALCULATOR:
                        transformed_table[2] = value
                return

            else:
                for i in range(len(transformed_table)):
                    self.update_transformed_table_helper(transformed_table[i], x, y, value, mode)

            return transformed_table

        elif mode == 2:
            if isinstance(transformed_table[0], tuple):
                if transformed_table[0] == (x, y) and (transformed_table[1] == TileType.FLOW_RATE_CALCULATOR or
                                                       transformed_table[
                                                           1] == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL or
                                                       transformed_table[1] == TileType.FLOW_RATE_CALCULATOR_VERTICAL):
                    transformed_table[2] = value
                return

            else:
                for i in range(len(transformed_table)):
                    self.update_transformed_table_helper(transformed_table[i], x, y, value, mode)

            return transformed_table

        else:
            raise Exception("Mode is not allowed")

    def update_transformed_table(self, transformed_table, entries, mode):
        # entries is a dictionary with keys as the location such as (x, y) and values as the values for that element
        # put this values to the transformed table instead of None's
        # be careful transformed table is a nested list
        # do it witt recursion

        updated_transformed_table = copy.deepcopy(transformed_table)

        for entry in entries:
            x, y = entry
            value = entries[entry]
            updated_transformed_table = self.update_transformed_table_helper(updated_transformed_table,
                                                                             x, y, value, mode)

        print("Updated Transformed Table:")
        print(f"\n{self.format_lines(updated_transformed_table)}")
        print("------------------------------------------------")

        return updated_transformed_table

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
