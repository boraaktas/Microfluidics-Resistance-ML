from typing import Optional

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


def build_3d_cell_maze(maze: np.ndarray,
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

    # height = height * 1.2 # Increase the height of the maze for 3D printing

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

    if len(corners) == 0:
        return maze_mesh

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

    # maze_mesh.show()

    return maze_mesh


def import_stl(cell_type_str: str,
               coming_direction: Optional[str],
               width: float,
               height: float,
               mesh_dict: dict) -> trimesh.Trimesh:

    file_cell_type_str = ""
    file_type_str = ""

    cell_direction_str = ""
    division_symmetric = ""

    if "START" in cell_type_str or "END" in cell_type_str:
        file_cell_type_str = "START_END"
        if "START" in cell_type_str:
            cell_direction_str = cell_type_str.split("_")[1]
        else:
            cell_direction_str = cell_type_str.split("_")[0]

    elif "DIVISION_3" in cell_type_str:
        file_cell_type_str = "DIVISION_3"
    elif "DIVISION_2" in cell_type_str:
        file_cell_type_str = "DIVISON_2"
        cell_direction_str = cell_type_str.split("_")[3]
        if cell_direction_str.lower() == coming_direction:
            file_type_str = "t1"
        else:
            file_type_str = "t2"
            if (cell_direction_str == "SOUTH" and coming_direction == "west") or (
                    cell_direction_str == "EAST" and coming_direction == "south") or (
                    cell_direction_str == "NORTH" and coming_direction == "east") or (
                    cell_direction_str == "WEST" and coming_direction == "north"):
                division_symmetric = "symmetric"
    elif "FLOW_RATE_CALCULATOR" in cell_type_str:
        file_cell_type_str = "FLOW_RATE_CALCULATOR"
        cell_direction_str = cell_type_str.split("_")[3]

        file_type_str = "v9"

    else:
        raise ValueError(f"Invalid cell type: {cell_type_str}")

    # height = height / 1.2  # Return height to previous value

    width_height_str = f"w{int(width * 100)}_h{int(height * 100)}"

    stl_key = ""
    if file_type_str == "":
        stl_key = f"{file_cell_type_str}_{width_height_str}.STL"
    else:
        if division_symmetric == "":
            stl_key = f"{file_cell_type_str}_{file_type_str}_{width_height_str}.STL"
        elif division_symmetric == "symmetric":
            stl_key = f"{file_cell_type_str}_{file_type_str}_{division_symmetric}_{width_height_str}.STL"

    # Load the mesh from the file path to the origin (0, 0, 0) of the world
    mesh = mesh_dict[stl_key]

    # Copy the mesh to avoid modifying the original
    mesh_rotated = mesh.copy()
    # Rotate the mesh to the correct orientation
    print(file_cell_type_str + " " + cell_direction_str)
    if file_cell_type_str == "START_END":
        if cell_direction_str == "NORTH":
            mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif cell_direction_str == "SOUTH":
            mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        elif cell_direction_str == "WEST":
            mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        elif cell_direction_str == "EAST":
            pass

    elif file_cell_type_str == "DIVISION_3":
        pass

    elif file_cell_type_str == "DIVISON_2":
        if file_type_str == "t1":
            if cell_direction_str == "EAST":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            elif cell_direction_str == "WEST":
                pass
            elif cell_direction_str == "SOUTH":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            elif cell_direction_str == "NORTH":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        elif file_type_str == "t2":
            if cell_direction_str == "NORTH" and coming_direction == "west":
                pass
            elif cell_direction_str == "WEST" and coming_direction == "south":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            elif cell_direction_str == "SOUTH" and coming_direction == "east":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            elif cell_direction_str == "EAST" and coming_direction == "north":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
            elif cell_direction_str == "NORTH" and coming_direction == "east" and division_symmetric == "symmetric":
                pass
            elif cell_direction_str == "EAST" and coming_direction == "south" and division_symmetric == "symmetric":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
            elif cell_direction_str == "SOUTH" and coming_direction == "west" and division_symmetric == "symmetric":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            elif cell_direction_str == "WEST" and coming_direction == "north" and division_symmetric == "symmetric":
                mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    elif file_cell_type_str == "FLOW_RATE_CALCULATOR":
        if cell_direction_str == "HORIZONTAL":
            pass
        elif cell_direction_str == "VERTICAL":
            mesh_rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    # Return the position before the rotation
    mesh_centroid = mesh.centroid

    if file_cell_type_str == "START_END":
        mesh_rotated_centroid = mesh.centroid
    else:
        mesh_rotated_centroid = mesh_rotated.centroid
    mesh_rotated.apply_translation(mesh_centroid - mesh_rotated_centroid)

    if file_cell_type_str == "START_END":
        if cell_direction_str == "NORTH":
            mesh_rotated.apply_translation([0, 0, 10])
        elif cell_direction_str == "SOUTH":
            mesh_rotated.apply_translation([10, 0, 0])
        elif cell_direction_str == "WEST":
            mesh_rotated.apply_translation([10, 0, 10])
        else:
            pass

    return mesh_rotated


def build_whole_circuit(DICT_FOR_3D_MODEL: dict,
                        mesh_dict: dict,
                        show_model: bool = False) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:

    # Constants
    base_height = 1
    base_side = 10

    # Read the dictionary and create a 3D model
    max_x = max([cell_loc[0] for cell_loc in DICT_FOR_3D_MODEL.keys()])
    max_y = max([cell_loc[1] for cell_loc in DICT_FOR_3D_MODEL.keys()])

    model_3d_dict = {}
    for i in range(max_x + 1):
        for j in range(max_y + 1):
            if (i, j) in DICT_FOR_3D_MODEL.keys():
                (cell_res_matrix, cell_fig,
                 pipe_width, pipe_height, pipe_fillet_radius,
                 cell_type, cell_coming_dir) = DICT_FOR_3D_MODEL[(i, j)]
                cell_type_str = str(cell_type).split(".")[1]

                if ("START" not in cell_type_str and "FLOW" not in cell_type_str and "END" not in cell_type_str and
                        "DIVISION" not in cell_type_str):

                    # Build a base under the maze_3d
                    base = trimesh.creation.box(extents=[base_side, base_side, base_height])
                    # Make the base white
                    base.visual.face_colors = [255, 255, 255, 255]

                    maze_3d = build_3d_cell_maze(maze=cell_res_matrix,
                                                 step_size_factor=0.5,
                                                 width=pipe_width,
                                                 height=pipe_height,
                                                 fillet_radius=pipe_fillet_radius)

                    maze_3d.apply_translation([(-base_side / 2), (-base_side / 2), (base_height / 2)])
                    maze_3d = trimesh.util.concatenate([maze_3d, base])

                    model_3d_dict[(i, j)] = maze_3d

                else:

                    imported_component = import_stl(cell_type_str=cell_type_str,
                                                    coming_direction=str(cell_coming_dir),
                                                    width=pipe_width,
                                                    height=pipe_height,
                                                    mesh_dict=mesh_dict)

                    imported_component.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2,
                                                                                               [1, 0, 0]))
                    imported_component.apply_translation([(-base_side / 2), (base_side / 2), (-base_height / 2)])

                    model_3d_dict[(i, j)] = imported_component

            elif (i, j) not in DICT_FOR_3D_MODEL.keys():

                # Build a base under the empty_cell
                base = trimesh.creation.box(extents=[base_side, base_side, base_height])
                # Make the base white
                base.visual.face_colors = [255, 255, 255, 255]

                model_3d_dict[(i, j)] = base

    # Scaled the keys of the model_3d_dict
    for key in model_3d_dict.keys():
        model_3d_dict[key] = (model_3d_dict[key], key[0] * base_side, key[1] * base_side)

    # Move the every component to the key coordinates
    for key in model_3d_dict.keys():
        model_3d_dict[key][0].apply_translation([model_3d_dict[key][1], model_3d_dict[key][2], 0])

    # Combine all the components
    combined_model = trimesh.Trimesh()
    for key in model_3d_dict.keys():
        combined_model = trimesh.util.concatenate([combined_model, model_3d_dict[key][0]])

    # Delete the base and replace it with the white base
    plane_origin = np.array([0, 0, 0.5])  # point on the plane
    plane_normal = np.array([0, 0, 1])  # normal vector of the plane (z direction)

    # Slice the mesh using the plane
    combined_model = combined_model.slice_plane(plane_origin, plane_normal)

    # Adding the Base Part
    bottom_base_width_constant = (max_x + 1.1)
    bottom_base_height_constant = (max_y + 1.1)

    bottom_base = trimesh.creation.box(extents=[(bottom_base_width_constant) * base_side, (bottom_base_height_constant) * base_side, base_height])
    bottom_base.visual.face_colors = [255, 255, 255, 255]
    bottom_base.apply_translation([max_x * 5, max_y * 5, 0])

    combined_model_with_base = trimesh.util.concatenate([combined_model, bottom_base])

    # Building the walls
    small_wall_height = 0.15
    small_wall_thickness = 0.5
    small_inside_wall = trimesh.creation.box(
        extents=[((bottom_base_width_constant) * base_side), ((bottom_base_height_constant) * base_side), (base_height + small_wall_height)])
    small_outside_wall = trimesh.creation.box(
        extents=[((bottom_base_width_constant) * base_side + small_wall_thickness), ((bottom_base_height_constant) * base_side + small_wall_thickness),
                 (base_height + small_wall_height)])
    small_wall = small_outside_wall.difference(small_inside_wall)
    small_wall.apply_translation(
        [max_x * 5, max_y * 5, abs(base_height / 2 - (base_height + small_wall_height) / 2)])

    dist_big_small = 2.5
    bottom_outer = trimesh.creation.box(extents=[((bottom_base_width_constant) * base_side + small_wall_thickness + dist_big_small),
                                                 ((bottom_base_height_constant) * base_side + small_wall_thickness + dist_big_small),
                                                 base_height])
    bottom_inner = trimesh.creation.box(
        extents=[((bottom_base_width_constant) * base_side + small_wall_thickness), ((bottom_base_height_constant) * base_side + small_wall_thickness),
                 base_height])
    bottom = bottom_outer.difference(bottom_inner)
    bottom.visual.face_colors = [255, 255, 255, 255]
    bottom.apply_translation([max_x * 5, max_y * 5, 0])

    big_wall_height = 2.4
    big_wall_thickness = 1
    big_inside_wall = trimesh.creation.box(
        extents=[((bottom_base_width_constant) * base_side + small_wall_thickness + dist_big_small),
                 ((bottom_base_height_constant) * base_side + small_wall_thickness + dist_big_small),
                 (base_height + big_wall_height)])
    big_outside_wall = trimesh.creation.box(
        extents=[((bottom_base_width_constant) * base_side + small_wall_thickness + dist_big_small + big_wall_thickness),
                 ((bottom_base_height_constant) * base_side + small_wall_thickness + dist_big_small + big_wall_thickness),
                 (base_height + big_wall_height)])
    big_wall = big_outside_wall.difference(big_inside_wall)
    big_wall.apply_translation([max_x * 5, max_y * 5, abs(base_height / 2 - (base_height + big_wall_height) / 2)])

    walls = trimesh.util.concatenate([small_wall, bottom, big_wall])

    # Combine the walls with the combined_model_with_base
    combined_model_with_base = trimesh.util.concatenate([combined_model_with_base, walls])

    biggest_bottom_box = trimesh.creation.box(
        extents=[(bottom_base_width_constant) * base_side + small_wall_thickness + dist_big_small + big_wall_thickness,
                 (bottom_base_height_constant) * base_side + small_wall_thickness + dist_big_small + big_wall_thickness,
                 1.4])
    biggest_bottom_box.apply_translation([max_x * 5, max_y * 5, -1.2])
    combined_model_with_base = trimesh.util.concatenate([combined_model_with_base, biggest_bottom_box])

    if show_model:
        combined_model.show(resolution=(1024, 768))

    return combined_model, combined_model_with_base
