import numpy as np
import trimesh
from trimesh.creation import box, cylinder
from src.maze_functions import get_coord_list_plot


def create_box_between_points(start, end, width, height):
    """
    Creates a box mesh between two points in 3D space.
    :param start: The starting point of the box (x, y, z).
    :param end: The ending point of the box (x, y, z).
    :param width: The width of the box.
    :param height: The height of the box.
    :return: A trimesh object representing the box.
    """
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    length = np.linalg.norm(vector)
    direction = vector / length

    # Create box along the z-axis
    box_mesh = box(extents=[height, width, length])

    # Align box with the direction vector
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    box_mesh.apply_transform(transform)

    # Move box to the correct start position
    translation = start + direction * (length / 2)
    box_mesh.apply_translation(translation)

    return box_mesh


def create_cylinder_between_points(start, end, radius, direction):
    """
    Creates a quarter cylinder mesh between two points in 3D space.
    :param start: The starting point of the cylinder (x, y, z).
    :param end: The ending point of the cylinder (x, y, z).
    :param radius: The radius of the cylinder.
    :param direction: The direction of the quarter cylinder ('NE', 'SE', 'NW', 'SW').
    :return: A trimesh object representing the quarter cylinder.
    """
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    length = np.linalg.norm(vector)
    direction_vector = vector / length

    # Create a full cylinder along the z-axis
    full_cylinder = cylinder(radius=radius, height=length)

    # Create a box mask to cut out three-quarters of the cylinder
    mask = box(extents=[2 * radius, 2 * radius, length])

    # Position the mask to cut out the correct quarter
    if direction == 'NE':
        mask.apply_translation([-radius, -radius, 0])
    elif direction == 'SE':
        mask.apply_translation([radius, radius, 0])
        mask.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    elif direction == 'NW':
        mask.apply_translation([radius, radius, 0])
        mask.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1]))
    elif direction == 'SW':
        mask.apply_translation([-radius, -radius, 0])
        mask.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))

    quarter_cylinder = full_cylinder.intersection(mask)

    # Align the quarter cylinder with the direction vector
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction_vector)
    quarter_cylinder.apply_transform(transform)

    # Move the quarter cylinder to the correct start position
    translation = start + direction_vector * (length / 2)
    quarter_cylinder.apply_translation(translation)

    return quarter_cylinder


def build_3d_maze(maze: np.ndarray,
                  step_size_factor: float,
                  width: float,
                  height: float,
                  fillet_radius: float,
                  ) -> trimesh.Trimesh:
    """
    This function builds a 3D model of the maze.
    :param maze: A 2D numpy array representing the maze where 0s are open paths, -1s are walls,
                positive numbers are steps in the path, and -2 is the target.
    :param step_size_factor: A float representing the step size factor. It is used to scale the step size.
    :param width: A float representing the width of the pipe.
    :param height: A float representing the height of the pipe.
    :param fillet_radius: A float representing the fillet radius of the pipe.
    :return: A trimesh object representing the 3D model of the maze.
    """
    maze = np.array(maze)
    coords_list = get_coord_list_plot(maze)

    # Multiply each coordinate by the step size factor
    scaled_coords = [(x * step_size_factor, y * step_size_factor, 0) for x, y, _ in coords_list]

    # From scaled coords, find the corners and save their turning direction
    corners_info = {}
    for i in range(1, len(scaled_coords) - 1):
        x1, y1, _ = scaled_coords[i - 1]
        x3, y3, _ = scaled_coords[i + 1]
        if x1 != x3 and y1 != y3:
            # Find the turning direction
            corner_x, corner_y, _ = scaled_coords[i]
            corner = np.array([corner_x, corner_y, 0])
            previous_point = np.array([x1, y1, 0])
            next_point = np.array([x3, y3, 0])
            origin = corner
            avg_point = (previous_point + next_point) / 2
            if avg_point[0] > origin[0] and avg_point[1] > origin[1]:
                direction = 'NE'
            elif avg_point[0] > origin[0] and avg_point[1] < origin[1]:
                direction = 'SE'
            elif avg_point[0] < origin[0] and avg_point[1] > origin[1]:
                direction = 'NW'
            else:
                direction = 'SW'

            corners_info[(corner_x, corner_y, 0)] = direction

    corners = list(corners_info.keys())
    # Create boxes (rectangles) for each segment of the path
    path_meshes = []
    for i in range(len(scaled_coords) - 1):
        start = scaled_coords[i]
        end = scaled_coords[i + 1]
        rect = create_box_between_points(start, end, width, height)
        path_meshes.append(rect)

    # Combine all path segments into a single mesh
    maze_mesh = trimesh.util.concatenate(path_meshes)

    # Shift the maze to the correct position
    maze_mesh.apply_translation([0, 0, height / 2])

    # Clear the corner areas to make space for the fillets
    inner_radius = (fillet_radius - width / 2)
    outer_radius = inner_radius + width
    corner_meshes = []
    for corner in corners:
        corner_center_top = corner + np.array([0, 0, height])
        corner_box = create_box_between_points(corner, corner_center_top, outer_radius, outer_radius)
        corner_meshes.append(corner_box)

    # Combine all corner segments into a single mesh
    corner_mesh = trimesh.util.concatenate(corner_meshes)

    # Subtract the corner areas from the maze mesh
    maze_mesh = maze_mesh.difference(corner_mesh)

    # Create fillets for the corners
    fillet_meshes = []
    for corner in corners:
        fillet_direction = corners_info[corner]
        shift = fillet_radius
        if fillet_direction == 'NE':
            fillet_center = corner + np.array([shift, shift, 0])
        elif fillet_direction == 'SE':
            fillet_center = corner + np.array([shift, -shift, 0])
        elif fillet_direction == 'NW':
            fillet_center = corner + np.array([-shift, shift, 0])
        else:
            fillet_center = corner + np.array([-shift, -shift, 0])

        fillet_center_top = fillet_center + np.array([0, 0, height])
        fillet_outside_cylinder = create_cylinder_between_points(fillet_center, fillet_center_top, outer_radius,
                                                                 fillet_direction)
        fillet_inside_cylinder = create_cylinder_between_points(fillet_center, fillet_center_top, inner_radius,
                                                                fillet_direction)
        fillet_mesh = fillet_outside_cylinder.difference(fillet_inside_cylinder)
        fillet_meshes.append(fillet_mesh)

    # Combine all fillet segments into a single mesh
    fillet_mesh = trimesh.util.concatenate(fillet_meshes)

    # Combine the maze mesh with the fillet mesh
    maze_mesh = trimesh.util.concatenate([maze_mesh, fillet_mesh])

    # Import the stl base from the folder
    base = trimesh.load_mesh('data/STL/base.STL')

    # Change the color of the base to white
    base.visual.face_colors = [255, 255, 255, 255]

    # Rotate the base 90 degrees around the x-axis and shift it up
    base.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    base.apply_translation([0, 10, height])

    # Combine the maze mesh with the base
    maze_mesh.apply_translation([0, 0, 0])
    maze_mesh = trimesh.util.concatenate([base, maze_mesh])

    maze_mesh.show()

    return maze_mesh


def main():
    maze = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 7, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 5, 6, 13, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 4, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [1, 2, 3, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 89],
        [-1, 0, 0, 0, 16, 17, 0, 0, 0, 0, 0, 33, 34, 0, 0, 0, 0, 0, 86, 87, -1],
        [-1, 0, 0, 0, 0, 18, 0, 0, 0, 30, 31, 32, 35, 36, 37, 0, 0, 0, 85, 0, -1],
        [-1, 0, 0, 0, 0, 19, 0, 0, 28, 29, 0, 0, 0, 0, 38, 39, 0, 0, 84, 83, -1],
        [-1, 0, 0, 0, 0, 20, 0, 26, 27, 0, 0, 0, 0, 0, 0, 40, 41, 78, 79, 82, -1],
        [-1, 0, 0, 0, 0, 21, 22, 25, 0, 0, 0, 49, 48, 47, 46, 45, 42, 77, 80, 81, -1],
        [-1, 0, 0, 0, 0, 0, 23, 24, 0, 54, 53, 50, 61, 62, 63, 44, 43, 76, 75, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 55, 52, 51, 60, 0, 64, 0, 72, 73, 74, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 56, 57, 58, 59, 66, 65, 70, 71, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 68, 69, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])

    # Define parameters
    step_size_factor = 0.5
    width = 0.05
    height = 0.05
    fillet_radius = 0.04

    maze_mesh = build_3d_maze(maze, step_size_factor, width, height, fillet_radius)
    maze_mesh.show()

    # Save the mesh to file
    # maze_mesh.export('maze.stl')


if __name__ == "__main__":
    main()
