import numpy as np
import trimesh


def build_3d_maze(maze, width, height, fillet_radius):
    maze = np.array(maze)
    maze_height, maze_width = maze.shape

    all_boxes = []
    for y in range(maze_height):
        for x in range(maze_width):
            if maze[y, x] != -1 and maze[y, x] != 0:
                box = trimesh.creation.box(extents=[width, width, height])
                box.apply_translation([x * width, y * height, height / 2])
                all_boxes.append(box)
            # if it is -1, it is a wall build with red
            if maze[y, x] == -1:
                box = trimesh.creation.box(extents=[width, width, height])
                box.apply_translation([x * width, y * height, height / 2])
                box.visual.face_colors = [255, 0, 0, 255]
                all_boxes.append(box)

    combined = trimesh.util.concatenate(all_boxes)
    # combined.export('maze.stl')

    return combined


def main():

    MAZE = [
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
    ]

    # Define parameters
    step_size_factor = 0.5
    width = 0.05
    height = 0.05
    fillet_radius = 0.04

    maze_mesh = build_3d_maze(MAZE, width, height, fillet_radius)

    # Show the 3D model in an interactive viewer
    maze_mesh.show()


if __name__ == "__main__":
    main()
