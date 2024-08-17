import numpy as np
from matplotlib import pyplot as plt

from src.maze_functions import get_coord_list_plot


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


def pretty_print_maze(maze: np.ndarray):
    for row in maze:
        print(" ".join(str(cell).rjust(4) for cell in row))