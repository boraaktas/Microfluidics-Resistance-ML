import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from feature_extractor import extract_features


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


def random_maze_generator(step_size, side_length, target_loc_mode: str):
    ratio = int(side_length / step_size)

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

    # Define the Start and End
    center = int((len_matrix - 1) / 2)
    state_matrix[center][0] = 1

    if not (target_loc_mode == "north" or target_loc_mode == "east"):
        raise ValueError("target_loc_mode must be either 'north' or 'east'")

    if target_loc_mode == "north":
        state_matrix[1][center] = -2  # north
    if target_loc_mode == "east":
        state_matrix[center][len_matrix - 2] = -2

    minus_two_coords = np.where(state_matrix == -2)
    target_coords = (minus_two_coords[0][0].item(), minus_two_coords[1][0].item())

    reached_target = False
    max_num = 1
    while not reached_target:
        max_num = np.max(state_matrix)
        max_coords = np.where(state_matrix == max_num)
        current_coords = (max_coords[0][0].item(), max_coords[1][0].item())

        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current_coords[0] + dx, current_coords[1] + dy
            if state_matrix[nx, ny] == 0 or (nx, ny) == target_coords:
                neighbours.append((nx, ny))

        corrected_neighbours = []
        for nx, ny in neighbours:
            if search_path(nx, ny, [], state_matrix, target_coords):
                corrected_neighbours.append((nx, ny))

        if len(corrected_neighbours) == 0:
            break

        next_coords = corrected_neighbours[np.random.randint(0, len(corrected_neighbours))]
        state_matrix[next_coords] = max_num + 1

        if next_coords == target_coords:
            reached_target = True

    # according to the target_loc_mode, we need to set the target to the last cell
    if target_loc_mode == "north":
        state_matrix[0][center] = max_num + 2
    else:
        state_matrix[center][len_matrix - 1] = max_num + 2

    # print state_matrix with a pretty format to the console
    for row in state_matrix:
        print(" ".join(str(cell).rjust(5) for cell in row))

    return state_matrix


def plot_maze(maze: np.ndarray):
    """
    This function plots the maze using lines to represent walls and paths.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    rows, cols = maze.shape

    coords: List[Tuple[int, int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if maze[i, j] > 0:
                coords.append((i, j, maze[i, j].item()))

    # sort the coords by the third element
    coords.sort(key=lambda x: x[2])

    # remove third element
    coords = [(x, y) for x, y, _ in coords]

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        ax.plot([y1, y2], [x1, x2], 'k')

    corners = [(0, 0), (rows - 1, 0), (rows - 1, cols - 1), (0, cols - 1)]
    # plot the walls with red borders line
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        if maze[x1, y1] == -1:
            ax.plot([y1, y2], [x1, x2], 'r')

    # set the aspect of the plot to be equal
    ax.set_aspect('equal')
    ax.axis('off')

    # get symmetric for the y-axis
    ax.invert_yaxis()

    plt.show()


if __name__ == "__main__":

    STEP_SIZE_FACTOR = 0.5
    STEP_SIZE = 1
    SIDE_LENGTH = 20
    TARGET_LOC_MODE = "east"
    WIDTH, HEIGHT, FILLET_RADIUS = 40, 40, 25

    MAZE = random_maze_generator(STEP_SIZE, SIDE_LENGTH, TARGET_LOC_MODE)
    plot_maze(MAZE)

    features = extract_features(MAZE, STEP_SIZE_FACTOR, WIDTH, HEIGHT, FILLET_RADIUS)
    print(features)
