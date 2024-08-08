from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from .feature_extractor import get_coord_list_matrix, get_coord_list_plot


def is_valid_move(x: int, y: int, visited, maze, target_coords):
    """Check if the move is valid (within bounds and not a wall)."""
    is_available = (x, y) not in visited and maze[x, y].item() == 0
    is_target = (x, y) == target_coords
    return is_available or is_target


def search_path(x: int, y: int, visited: List[Tuple[int, int]],
                maze: np.ndarray, target_coords: Tuple[int, int]
                ) -> bool:
    """Recursive function to search for the path."""
    if (x, y) == target_coords:
        return True

    # Mark the current cell as visited
    visited.append((x, y))

    neighbours = []
    # Explore neighbors
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if is_valid_move(nx, ny, visited, maze, target_coords):
            neighbours.append((nx, ny))

    # sort neighbors by Euclidean distance to the target
    neighbours.sort(key=lambda x_i: (x_i[0] - target_coords[0]) ** 2 + (x_i[1] - target_coords[1]) ** 2)

    for nx, ny in neighbours:
        if (nx, ny) == target_coords:
            return True

    for nx, ny in neighbours:
        if search_path(nx, ny, visited, maze, target_coords):
            return True

    return False


def create_empty_maze(side_length: int,
                      target_loc_mode: str) -> np.ndarray:
    if target_loc_mode not in ["east", "north"]:
        raise ValueError("target_loc_mode should be either 'east' or 'north'")

    ratio = int(side_length / 1)

    state_matrix = np.zeros((ratio + 2, ratio + 2), dtype=np.int32)
    if (ratio % 2) == 0:
        state_matrix = np.zeros((ratio + 1, ratio + 1), dtype=np.int32)

    # Define the Walls
    len_matrix = state_matrix.shape[0]
    for i in range(len_matrix):
        for j in range(len_matrix):
            if i == 0 or i == len_matrix - 1:
                state_matrix[i][j] = -1
            if j == 0 or j == len_matrix - 1:
                state_matrix[i][j] = -1

    return state_matrix


def complete_maze(maze: np.ndarray,
                  start_coords: List[int],
                  target_coords: Tuple[int, int],
                  coords_list: List[List[int]],  # every inner list contains (x, y, value)
                  path_finding_mode: str):
    if path_finding_mode not in ["shortest", "longest", "random"]:
        raise ValueError("path_finding_mode should be either 'shortest', 'longest' or 'random'")

    reached_target = False
    current_num = start_coords[2]
    current_coords = (start_coords[0], start_coords[1])
    maze[current_coords] = current_num

    while not reached_target:

        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current_coords[0] + dx, current_coords[1] + dy
            if maze[nx, ny] == 0 or (nx, ny) == target_coords:
                neighbours.append((nx, ny))

        corrected_neighbours = []
        for nx, ny in neighbours:
            if search_path(nx, ny, [], maze, target_coords):
                corrected_neighbours.append((nx, ny))

        if len(corrected_neighbours) == 0:
            break

        corrected_neighbors_probs = []
        total_dist = 0
        for nx, ny in corrected_neighbours:
            current_dist = (nx - target_coords[0]) ** 2 + (ny - target_coords[1]) ** 2
            corrected_neighbors_probs.append(current_dist)
            total_dist += current_dist

        # normalize the probabilities
        corrected_neighbors_probs = [x / total_dist for x in corrected_neighbors_probs] if total_dist > 0 else [1]

        # choose the minimum probability
        sorted_probs = np.sort(corrected_neighbors_probs)

        if path_finding_mode == "shortest":
            chosen_prob = sorted_probs[0]
        elif path_finding_mode == "longest":
            chosen_prob = sorted_probs[-1]
        else:
            chosen_prob = np.random.choice(sorted_probs)

        next_coords = corrected_neighbours[corrected_neighbors_probs.index(chosen_prob)]

        maze[next_coords] = current_num + 1
        current_num += 1
        current_coords = next_coords

        if next_coords == target_coords:
            reached_target = True
            # update other coordinates after the target location
            just_coords = [(x, y) for x, y, _ in coords_list]
            if target_coords in just_coords:
                target_index = just_coords.index(target_coords)
                current_num = maze[target_coords]
                for i in range(target_index, len(coords_list)):
                    x, y, _ = coords_list[i]
                    maze[x, y] = current_num
                    current_num += 1

    return maze


def random_maze_generator(side_length, target_loc_mode: str,
                          path_finding_mode: str = "random") -> np.ndarray:
    random_matrix = create_empty_maze(side_length, target_loc_mode)
    len_matrix = random_matrix.shape[0]
    center = int((len_matrix - 1) / 2)

    start_coords = [center, 0, 1]
    target_coords = (0, center) if target_loc_mode == "north" else (center, len_matrix - 1)
    coords_list = get_coord_list_matrix(random_matrix)

    random_matrix = complete_maze(random_matrix,
                                  start_coords,
                                  target_coords,
                                  coords_list,
                                  path_finding_mode)

    return random_matrix


def plot_maze(maze: np.ndarray,
              print_maze: bool = False,
              show_plot: bool = True,
              ) -> plt.Figure:
    """
    This function plots the maze using lines to represent walls and paths.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    :param print_maze: A boolean that indicates whether to print the maze to the console or not.
    :param show_plot: A boolean that indicates whether to show the plot or not.
    :return: A matplotlib figure representing
    """
    # print state_matrix with a pretty format to the console
    if print_maze:
        for row in maze:
            print(" ".join(str(cell).rjust(5) for cell in row))

    fig, ax = plt.subplots(figsize=(10, 10))
    rows, cols = maze.shape

    coords = get_coord_list_plot(maze)

    # sort the coords by the third element
    coords.sort(key=lambda x: x[2])

    # remove third element
    coords = [(x, y) for x, y, _ in coords]

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        # if they are neighbors, plot a line
        if abs(x1 - x2) + abs(y1 - y2) == 1:
            ax.plot([x1, x2], [y1, y2], 'k')

    corners = [(0, 0), (rows - 1, 0), (rows - 1, cols - 1), (0, cols - 1)]
    # plot the walls with red borders line
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        if maze[x1, y1] == -1:
            ax.plot([x1, x2], [y1, y2], 'r')

    # set the aspect of the plot to be equal
    ax.set_aspect('equal')
    ax.axis('off')

    # decrease the white space around the plot
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # increase the line width
    for line in ax.lines:
        line.set_linewidth(10)

    if show_plot:
        plt.show()

    # return plt as image
    return fig


def plot_other_components(component_type: str,
                          shape: tuple[int, int],
                          show_plot: bool = True) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(10, 10))
    rows, cols = shape

    middle_x = int(rows / 2)
    middle_y = int(cols / 2)

    # first plot the walls
    ax.plot([0, 0], [0, cols], 'r-')
    ax.plot([0, rows], [cols, cols], 'r-')
    ax.plot([rows, rows], [cols, 0], 'r-')
    ax.plot([rows, 0], [0, 0], 'r-')

    if component_type == "DIVISION_3":
        ax.plot([0, rows], [middle_y, middle_y], 'k')
        ax.plot([middle_x, middle_x], [middle_y, cols], 'k')
        ax.plot([middle_x, middle_x], [middle_y, 0], 'k')

    elif component_type == "DIVISION_2_FROM_NORTH":
        ax.plot([0, rows], [middle_y, middle_y], 'k-')
        ax.plot([middle_x, middle_x], [middle_y, cols], 'k')

    elif component_type == "DIVISION_2_FROM_SOUTH":
        ax.plot([0, rows], [middle_y, middle_y], 'k')
        ax.plot([middle_x, middle_x], [middle_y, 0], 'k')

    elif component_type == "DIVISION_2_FROM_WEST":
        ax.plot([0, middle_y], [middle_y, middle_y], 'k')
        ax.plot([middle_x, middle_x], [0, cols], 'k')

    elif component_type == "DIVISION_2_FROM_EAST":
        ax.plot([cols, middle_y], [middle_y, middle_y], 'k')
        ax.plot([middle_x, middle_x], [cols, 0], 'k')

    elif component_type == "FLOW_RATE_CALCULATOR_HORIZONTAL":
        ax.plot(middle_x, middle_y, 'ko', markersize=300)
        ax.plot([0, rows], [middle_y, middle_y], 'k')

    elif component_type == "FLOW_RATE_CALCULATOR_VERTICAL":
        ax.plot(middle_x, middle_y, 'ko', markersize=300)
        ax.plot([middle_x, middle_x], [0, cols], 'k')

    elif component_type == "START_EAST":
        ax.plot(middle_x, middle_y, 'go', markersize=300)
        ax.plot([middle_x, rows], [middle_y, middle_y], 'k', zorder=1)

    elif component_type == "START_WEST":
        ax.plot(middle_x, middle_y, 'go', markersize=300)
        ax.plot([middle_x, 0], [middle_y, middle_y], 'k', zorder=1)

    elif component_type == "START_NORTH":
        ax.plot(middle_x, middle_y, 'go', markersize=300)
        ax.plot([middle_x, middle_x], [middle_y, cols], 'k', zorder=1)

    elif component_type == "START_SOUTH":
        ax.plot(middle_x, middle_y, 'go', markersize=300)
        ax.plot([middle_x, middle_x], [middle_y, 0], 'k', zorder=1)

    elif component_type == "EAST_END":
        ax.plot(middle_x, middle_y, 'ro', markersize=300)
        ax.plot([middle_x, rows], [middle_y, middle_y], 'k', zorder=1)

    elif component_type == "WEST_END":
        ax.plot(middle_x, middle_y, 'ro', markersize=300)
        ax.plot([middle_x, 0], [middle_y, middle_y], 'k', zorder=1)

    elif component_type == "NORTH_END":
        ax.plot(middle_x, middle_y, 'ro', markersize=300)
        ax.plot([middle_x, middle_x], [middle_y, cols], 'k', zorder=1)

    elif component_type == "SOUTH_END":
        ax.plot(middle_x, middle_y, 'ro', markersize=300)
        ax.plot([middle_x, middle_x], [middle_y, 0], 'k', zorder=1)

    elif component_type == "EMPTY":
        pass

    # set the aspect of the plot to be equal
    ax.set_aspect('equal')
    ax.axis('off')

    # decrease the white space around the plot
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # increase the line width
    for line in ax.lines:
        line.set_linewidth(10)

    if show_plot:
        plt.show()

    return fig
