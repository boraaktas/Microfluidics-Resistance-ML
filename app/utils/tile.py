from .tile_type import TileType


class Tile:

    def __init__(self, tile_type: TileType = TileType.EMPTY):
        self.tile_type = tile_type

    def set_tile_type(self, tile_type: TileType):
        print(f"{self.tile_type} -> {tile_type}")
        self.tile_type = tile_type

    def rotate_tile(self):
        if self.tile_type == TileType.EMPTY:
            self.tile_type = TileType.EMPTY
        if self.tile_type == TileType.START_EAST:
            self.tile_type = TileType.START_NORTH
        elif self.tile_type == TileType.START_NORTH:
            self.tile_type = TileType.START_WEST
        elif self.tile_type == TileType.START_WEST:
            self.tile_type = TileType.START_SOUTH
        elif self.tile_type == TileType.START_SOUTH:
            self.tile_type = TileType.START_EAST
        elif self.tile_type == TileType.EAST_END:
            self.tile_type = TileType.NORTH_END
        elif self.tile_type == TileType.NORTH_END:
            self.tile_type = TileType.WEST_END
        elif self.tile_type == TileType.WEST_END:
            self.tile_type = TileType.SOUTH_END
        elif self.tile_type == TileType.SOUTH_END:
            self.tile_type = TileType.EAST_END
        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL:
            self.tile_type = TileType.FLOW_RATE_CALCULATOR_VERTICAL
        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_VERTICAL:
            self.tile_type = TileType.FLOW_RATE_CALCULATOR_HORIZONTAL
        elif self.tile_type == TileType.STRAIGHT_HORIZONTAL:
            self.tile_type = TileType.STRAIGHT_VERTICAL
        elif self.tile_type == TileType.STRAIGHT_VERTICAL:
            self.tile_type = TileType.STRAIGHT_HORIZONTAL
        elif self.tile_type == TileType.TURN_WEST_NORTH:
            self.tile_type = TileType.TURN_WEST_SOUTH
        elif self.tile_type == TileType.TURN_WEST_SOUTH:
            self.tile_type = TileType.TURN_EAST_SOUTH
        elif self.tile_type == TileType.TURN_EAST_SOUTH:
            self.tile_type = TileType.TURN_EAST_NORTH
        elif self.tile_type == TileType.TURN_EAST_NORTH:
            self.tile_type = TileType.TURN_WEST_NORTH
        elif self.tile_type == TileType.DIVISION_2_FROM_EAST:
            self.tile_type = TileType.DIVISION_2_FROM_NORTH
        elif self.tile_type == TileType.DIVISION_2_FROM_NORTH:
            self.tile_type = TileType.DIVISION_2_FROM_WEST
        elif self.tile_type == TileType.DIVISION_2_FROM_WEST:
            self.tile_type = TileType.DIVISION_2_FROM_SOUTH
        elif self.tile_type == TileType.DIVISION_2_FROM_SOUTH:
            self.tile_type = TileType.DIVISION_2_FROM_EAST
        elif self.tile_type == TileType.DIVISION_3:
            self.tile_type = TileType.DIVISION_3
        else:
            raise Exception("Tile type not found")

    def find_going_direction(self, coming_direction: str):
        if self.tile_type == TileType.EMPTY:
            return ["empty"]

        if self.tile_type == TileType.START_EAST and coming_direction == "start":
            return ["east"]
        elif self.tile_type == TileType.START_WEST and coming_direction == "start":
            return ["west"]
        elif self.tile_type == TileType.START_NORTH and coming_direction == "start":
            return ["north"]
        elif self.tile_type == TileType.START_SOUTH and coming_direction == "start":
            return ["south"]

        elif self.tile_type == TileType.EAST_END and coming_direction == "east":
            return ["end"]
        elif self.tile_type == TileType.WEST_END and coming_direction == "west":
            return ["end"]
        elif self.tile_type == TileType.NORTH_END and coming_direction == "north":
            return ["end"]
        elif self.tile_type == TileType.SOUTH_END and coming_direction == "south":
            return ["end"]

        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL and coming_direction == "east":
            return ["west"]
        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_HORIZONTAL and coming_direction == "west":
            return ["east"]

        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_VERTICAL and coming_direction == "north":
            return ["south"]
        elif self.tile_type == TileType.FLOW_RATE_CALCULATOR_VERTICAL and coming_direction == "south":
            return ["north"]

        elif self.tile_type == TileType.STRAIGHT_HORIZONTAL and coming_direction == "east":
            return ["west"]
        elif self.tile_type == TileType.STRAIGHT_HORIZONTAL and coming_direction == "west":
            return ["east"]

        elif self.tile_type == TileType.STRAIGHT_VERTICAL and coming_direction == "north":
            return ["south"]
        elif self.tile_type == TileType.STRAIGHT_VERTICAL and coming_direction == "south":
            return ["north"]

        elif self.tile_type == TileType.TURN_WEST_NORTH and coming_direction == "west":
            return ["north"]
        elif self.tile_type == TileType.TURN_WEST_NORTH and coming_direction == "north":
            return ["west"]

        elif self.tile_type == TileType.TURN_WEST_SOUTH and coming_direction == "west":
            return ["south"]
        elif self.tile_type == TileType.TURN_WEST_SOUTH and coming_direction == "south":
            return ["west"]

        elif self.tile_type == TileType.TURN_EAST_NORTH and coming_direction == "east":
            return ["north"]
        elif self.tile_type == TileType.TURN_EAST_NORTH and coming_direction == "north":
            return ["east"]

        elif self.tile_type == TileType.TURN_EAST_SOUTH and coming_direction == "east":
            return ["south"]
        elif self.tile_type == TileType.TURN_EAST_SOUTH and coming_direction == "south":
            return ["east"]

        elif self.tile_type == TileType.DIVISION_2_FROM_EAST and coming_direction == "east":
            return ["north", "south"]
        elif self.tile_type == TileType.DIVISION_2_FROM_EAST and coming_direction == "north":
            return ["east", "south"]
        elif self.tile_type == TileType.DIVISION_2_FROM_EAST and coming_direction == "south":
            return ["east", "north"]

        elif self.tile_type == TileType.DIVISION_2_FROM_NORTH and coming_direction == "north":
            return ["west", "east"]
        elif self.tile_type == TileType.DIVISION_2_FROM_NORTH and coming_direction == "west":
            return ["north", "east"]
        elif self.tile_type == TileType.DIVISION_2_FROM_NORTH and coming_direction == "east":
            return ["north", "west"]

        elif self.tile_type == TileType.DIVISION_2_FROM_WEST and coming_direction == "west":
            return ["north", "south"]
        elif self.tile_type == TileType.DIVISION_2_FROM_WEST and coming_direction == "north":
            return ["west", "south"]
        elif self.tile_type == TileType.DIVISION_2_FROM_WEST and coming_direction == "south":
            return ["west", "north"]

        elif self.tile_type == TileType.DIVISION_2_FROM_SOUTH and coming_direction == "south":
            return ["west", "east"]
        elif self.tile_type == TileType.DIVISION_2_FROM_SOUTH and coming_direction == "west":
            return ["south", "east"]
        elif self.tile_type == TileType.DIVISION_2_FROM_SOUTH and coming_direction == "east":
            return ["south", "west"]

        elif self.tile_type == TileType.DIVISION_3 and coming_direction == "north":
            return ["west", "east", "south"]
        elif self.tile_type == TileType.DIVISION_3 and coming_direction == "south":
            return ["west", "east", "north"]
        elif self.tile_type == TileType.DIVISION_3 and coming_direction == "west":
            return ["north", "south", "east"]
        elif self.tile_type == TileType.DIVISION_3 and coming_direction == "east":
            return ["north", "south", "west"]

        else:
            raise Exception(f"Coming direction: {coming_direction} is not allowed for tile type: {self.tile_type}")
