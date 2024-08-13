import numpy as np
from matplotlib import pyplot as plt

from src import PredictionModel, GenerativeModel
from src.maze_functions import plot_maze
from .tile_type import TileType


class ResistanceGenerator:

    def __init__(self: 'ResistanceGenerator',
                 prediction_model: PredictionModel):
        """
        Initializes the ResistanceGenerator object.
        """

        self.prediction_model = prediction_model

    def generate_resistance(self: 'ResistanceGenerator',
                            desired_resistance: float,
                            target_loc_mode: str,
                            width: float,
                            height: float,
                            fillet_radius: float,
                            step_size_factor: float,
                            side_length: float) -> np.ndarray:
        """
        Generates a cell resistance matrix for the given parameters.

        :param desired_resistance: The desired resistance value of the maze that is
                                   calculated by the optimization model.
        :param target_loc_mode: The location of the target. It can be either "east" or "north".
        :param width: The width of the pipe.
        :param height: The height of the pipe.
        :param fillet_radius: The radius of the fillet.
        :param step_size_factor: The step size factor of the path.
        :param side_length: The side length of the cell.

        :return: cell_resistance: The cell resistance matrix of the maze.
        """

        generative_model = GenerativeModel(prediction_model=self.prediction_model,
                                           desired_resistance=desired_resistance,
                                           step_size_factor=step_size_factor,
                                           width=width,
                                           height=height,
                                           fillet_radius=fillet_radius,
                                           target_loc_mode=target_loc_mode,
                                           method="TS",
                                           side_length=side_length)

        cell_resistance = generative_model.generate_maze()

        return cell_resistance

    @staticmethod
    def adjust_orientation(cell_resistance_matrix: np.ndarray,
                           cell_type: TileType):
        """
        Adjusts the orientation of the cell resistance matrix according to the target location.

        :param cell_resistance_matrix: The cell resistance matrix of the maze.
        :param cell_type: The location of the target. It can be either "east" or "north".
        :return: adjusted_cell_resistance: The adjusted cell resistance matrix of the maze.
        """

        cell_resistance_matrix_copy = cell_resistance_matrix.copy()

        # if cell_type = STRAIGHT_VERTICAL
        if cell_type == TileType.STRAIGHT_VERTICAL:
            # 90-degree rotation in counter-clockwise
            rotated_cell_resistance = np.rot90(cell_resistance_matrix_copy, 1)
        elif cell_type == TileType.TURN_WEST_SOUTH:
            # 90-degree rotation in counter-clockwise
            rotated_cell_resistance = np.rot90(cell_resistance_matrix_copy, k=1)
        elif cell_type == TileType.TURN_EAST_SOUTH:
            # 180-degree rotation in counter-clockwise
            rotated_cell_resistance = np.rot90(cell_resistance_matrix_copy, k=2)
        elif cell_type == TileType.TURN_EAST_NORTH:
            # 90-degree rotation in clockwise
            rotated_cell_resistance = np.rot90(cell_resistance_matrix_copy, k=3)
        elif cell_type == TileType.TURN_WEST_NORTH or cell_type == TileType.STRAIGHT_HORIZONTAL:
            rotated_cell_resistance = cell_resistance_matrix_copy
        else:
            raise ValueError("Invalid cell type.")

        return rotated_cell_resistance

    @staticmethod
    def get_rotated_matrix_and_figure(cell_resistance_matrix: np.ndarray,
                                      cell_type: TileType) -> tuple[np.ndarray, plt.Figure]:
        """
        Adjusts the orientation of the cell resistance matrix according to the target location.

        :param cell_resistance_matrix: The cell resistance matrix of the maze.
        :param cell_type: The location of the target. It can be either "east" or "north".
        :return: rotated_cell_resistance: The adjusted cell resistance matrix of the maze.
        :return: rotated_cell_resistance_image: The image of the adjusted cell resistance matrix.
        """

        rotated_cell_resistance = ResistanceGenerator.adjust_orientation(cell_resistance_matrix, cell_type)
        rotated_cell_resistance_fig = plot_maze(rotated_cell_resistance, show_plot=False)

        return rotated_cell_resistance, rotated_cell_resistance_fig
