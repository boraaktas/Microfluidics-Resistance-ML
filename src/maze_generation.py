from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from feature_extractor import extract_features, get_coord_list_matrix, get_coord_list_plot
from prediction_model import PredictionModel

from build_3D import build_3d_maze


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


def plot_maze(maze: np.ndarray, print_maze: bool = False):
    """
    This function plots the maze using lines to represent walls and paths.

    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                 positive numbers are steps in the path, and -2 is the target.
    :param print_maze: A boolean that indicates whether to print the maze to the console or not.
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

    plt.show()


if __name__ == "__main__":
    STEP_SIZE_FACTOR = 0.5
    SIDE_LENGTH = 20
    TARGET_LOC_MODE = "east"  # east or north
    WIDTH, HEIGHT, FILLET_RADIUS = 0.10, 0.10, 0.10
    PATH_FINDING_MODE = "random"

    MAZE = random_maze_generator(SIDE_LENGTH, TARGET_LOC_MODE, PATH_FINDING_MODE)
    plot_maze(MAZE, print_maze=True)

    FEATURES_DICT = extract_features(MAZE, STEP_SIZE_FACTOR, WIDTH, HEIGHT, FILLET_RADIUS)
    print(FEATURES_DICT)

    maze = build_3d_maze(maze=MAZE, step_size_factor=STEP_SIZE_FACTOR, width=WIDTH, height=HEIGHT, fillet_radius=FILLET_RADIUS)
    maze.show()

    PREDICTION_MODEL = PredictionModel(base_learners_pickle_path='../data/pickles/base_learner_pickles/',
                                       meta_learner_pickle_path='../data/pickles/meta_learner_pickles/')

    RANDOM_MAZE_RESISTANCE = PREDICTION_MODEL.predict(FEATURES_DICT)
    print(f"The prediction resistance of the random maze is: {RANDOM_MAZE_RESISTANCE}")

