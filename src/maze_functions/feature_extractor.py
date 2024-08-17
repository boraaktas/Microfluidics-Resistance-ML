from typing import List

import numpy as np


def get_coord_list_matrix(maze: np.ndarray) -> List[List[int]]:
    """
    This function takes a maze and returns a list of coordinates of the path.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    :return: A list of coordinates of the path.
    """
    rows, cols = maze.shape

    coords: List[List[int]] = []
    for i in range(rows):
        for j in range(cols):
            if maze[i, j] > 0:
                coords.append([i, j, maze[i, j].item()])

    # sort the coords by the third element
    coords.sort(key=lambda x: x[2])

    return coords


def get_coord_list_plot(maze: np.ndarray) -> List[List[int]]:
    """
    This function takes a maze and returns a list of coordinates of the path.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    :return: A list of coordinates of the path.
    """
    rows, cols = maze.shape

    coords: List[List[int]] = []
    for i in range(rows):
        for j in range(cols):
            if maze[i, j] > 0:
                coords.append([j, rows-i-1, maze[i, j].item()])

    # sort the coords by the third element
    coords.sort(key=lambda x: x[2])

    return coords


def get_total_length(maze_coords: List[List[int]], step_size_factor: float) -> float:
    """
    This function calculates the total length of the path in the maze.

    :param maze_coords: A list of coordinates of the path.
    :param step_size_factor: A float representing the step size factor. It is used to scale the step size.
    :return: A float representing the total length of the path in the maze.
    """
    total_length = len(maze_coords) * step_size_factor
    return total_length


def get_no_corners(maze_coords: List[List[int]]) -> int:
    """
    This function calculates the number of corners in the path of the maze.

    :param maze_coords: A list of coordinates of the path.
    :return: An integer representing the number of corners in the path of the maze.
    """
    no_corners = 0

    for i in range(1, len(maze_coords) - 1):
        x1, y1, v1 = maze_coords[i - 1]
        x3, y3, v3 = maze_coords[i + 1]
        if x1 != x3 and y1 != y3:
            no_corners += 1
    return no_corners


def get_coord_corners_list(maze_coords: List[List[int]]) -> List[List[int]]:
    """
    This function calculates the coordinates of the corners in the path of the maze.

    :param maze_coords: A list of coordinates of the path.
    :return: A list of coordinates of the corners in the path of the maze.
    """
    corners_coord = []

    for i in range(1, len(maze_coords) - 1):
        x1, y1, v1 = maze_coords[i - 1]
        x2, y2, v2 = maze_coords[i]
        x3, y3, v3 = maze_coords[i + 1]
        if x1 != x3 and y1 != y3:
            corners_coord.append([x2, y2, v2])
        else:
            corners_coord.append([-1, -1, -1])

    return corners_coord


def extract_features(maze: np.ndarray,
                     step_size_factor: float,
                     width: float,
                     height: float,
                     fillet_radius: float,
                     ) -> dict:
    """
    This function extracts features from the maze.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    :param step_size_factor: A float representing the step size factor. It is used to scale the step size.
    :param width: A float representing the width of the pipe.
    :param height: A float representing the height of the pipe.
    :param fillet_radius: A float representing the fillet radius of the pipe.

    :return: A dictionary containing the extracted features.
    """

    maze_coords = get_coord_list_matrix(maze)

    total_length = get_total_length(maze_coords, step_size_factor)
    no_corners = get_no_corners(maze_coords)

    feature_dict = {
        "Total_Length": total_length,
        "Corner": no_corners,
        "Width": width,
        "Height": height,
        "Fillet_Radius": fillet_radius
    }

    return feature_dict
