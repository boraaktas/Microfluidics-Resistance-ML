from .tile_type import TileType


class Constants:
    ALLOWED_COMING_DIRECTIONS = {
        TileType.START_EAST: ["start"],
        TileType.START_WEST: ["start"],
        TileType.START_NORTH: ["start"],
        TileType.START_SOUTH: ["start"],

        TileType.EAST_END: ["east"],
        TileType.WEST_END: ["west"],
        TileType.NORTH_END: ["north"],
        TileType.SOUTH_END: ["south"],

        TileType.FLOW_RATE_CALCULATOR_HORIZONTAL: ["east", "west"],
        TileType.FLOW_RATE_CALCULATOR_VERTICAL: ["north", "south"],

        TileType.STRAIGHT_HORIZONTAL: ["east", "west"],
        TileType.STRAIGHT_VERTICAL: ["north", "south"],

        TileType.TURN_WEST_NORTH: ["west", "north"],
        TileType.TURN_WEST_SOUTH: ["west", "south"],
        TileType.TURN_EAST_NORTH: ["east", "north"],
        TileType.TURN_EAST_SOUTH: ["east", "south"],

        TileType.DIVISION_2_FROM_EAST: ["east", "north", "south"],
        TileType.DIVISION_2_FROM_WEST: ["west", "north", "south"],
        TileType.DIVISION_2_FROM_NORTH: ["north", "west", "east"],
        TileType.DIVISION_2_FROM_SOUTH: ["south", "west", "east"],

        TileType.DIVISION_3: ["north", "south", "west", "east"]
    }

    STARTER_TYPES = [TileType.START_EAST, TileType.START_WEST, TileType.START_NORTH, TileType.START_SOUTH]

    END_TYPES = [TileType.EAST_END, TileType.WEST_END, TileType.NORTH_END, TileType.SOUTH_END]

    DIVISION_TYPES = [TileType.DIVISION_2_FROM_EAST, TileType.DIVISION_2_FROM_WEST,
                      TileType.DIVISION_2_FROM_NORTH, TileType.DIVISION_2_FROM_SOUTH,
                      TileType.DIVISION_3]

    OPPOSITE_DIRECTIONS: dict[str, str] = {
        "north": "south",
        "south": "north",
        "west": "east",
        "east": "west"
    }

    RL_TILES = {TileType.STRAIGHT_HORIZONTAL, TileType.STRAIGHT_VERTICAL}
    RC_TILES = {TileType.TURN_WEST_NORTH, TileType.TURN_WEST_SOUTH, TileType.TURN_EAST_NORTH, TileType.TURN_EAST_SOUTH}
    Q_TILES = {TileType.FLOW_RATE_CALCULATOR_HORIZONTAL,
               TileType.FLOW_RATE_CALCULATOR_VERTICAL}
    P_TILES = {
        TileType.START_EAST, TileType.START_WEST, TileType.START_NORTH, TileType.START_SOUTH,
        TileType.EAST_END, TileType.WEST_END, TileType.NORTH_END, TileType.SOUTH_END,
        TileType.DIVISION_2_FROM_EAST, TileType.DIVISION_2_FROM_WEST, TileType.DIVISION_2_FROM_NORTH,
        TileType.DIVISION_2_FROM_SOUTH, TileType.DIVISION_3
    }
