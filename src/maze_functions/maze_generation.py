from typing import Tuple, List

import numpy as np

from src.maze_functions import get_coord_list_matrix


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


def destruct_maze(maze: np.ndarray,
                  locations_in_maze: list[list[int]],
                  index_1: int,
                  index_2: int) -> tuple[np.ndarray, list[list[int]]]:

    sorted_indices = np.sort([index_1, index_2])

    deleted_locations_in_maze = locations_in_maze[sorted_indices[0]: sorted_indices[1]]

    deleted_maze = np.copy(maze)
    for X, Y, V in deleted_locations_in_maze:
        deleted_maze[X, Y] = 0

    return deleted_maze, deleted_locations_in_maze


def repair_maze(deleted_maze: np.ndarray,
                old_maze: np.ndarray,
                locations_in_maze: list[list[int]],
                random_indices: np.ndarray,
                repair_mode: str) -> np.ndarray:

    left_locations = get_coord_list_matrix(deleted_maze)

    last_loc_before_empty = locations_in_maze[random_indices[0] - 1]

    target_loc = (locations_in_maze[random_indices[1]][0], locations_in_maze[random_indices[1]][1])
    start_coords = [last_loc_before_empty[0], last_loc_before_empty[1],
                    old_maze[last_loc_before_empty[0], last_loc_before_empty[1]].item()]

    new_maze = complete_maze(deleted_maze,
                             start_coords,
                             target_loc,
                             left_locations,
                             repair_mode)

    return new_maze
