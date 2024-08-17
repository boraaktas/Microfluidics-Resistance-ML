import ttkbootstrap as ttk

from app.utils import TileType, Tile


class Menu_Section:
    def __init__(self, main_gui):
        self.main_gui = main_gui
        self.root = main_gui.root

        self.frame = ttk.Frame(self.root, borderwidth=2, relief="solid")
        self.frame.place(relx=0.7, relwidth=0.3, relheight=1, anchor='nw')

        self.tile_images = main_gui.images
        self.create_menu_tiles()

        self.table_obj = main_gui.table_obj

        build_button = ttk.Button(self.frame, text="Build", command=self.build)
        build_button.pack(pady=10)

    def create_menu_tiles(self):
        # ----------------- Tile 1 -----------------
        frame_1 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_1.pack(pady=10)

        tile_1 = Tile()
        tile_1.tile_type = TileType(1)

        image_label_1 = ttk.Label(frame_1, image=self.tile_images[1][0])
        image_label_1.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_1 = ttk.Button(frame_1,
                                     text="Rotate", command=lambda t=tile_1: self.rotate_tile(t, image_label_1))
        rotate_button_1.grid(row=0, column=1, padx=10)

        select_button_1 = ttk.Button(frame_1, text="Select", command=lambda t=tile_1: self.select_tile(t))
        select_button_1.grid(row=0, column=2, padx=10)

        # ----------------- Tile 5 -----------------
        frame_5 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_5.pack(pady=10)

        tile_5 = Tile()
        tile_5.tile_type = TileType(5)

        image_label_5 = ttk.Label(frame_5, image=self.tile_images[5][0])
        image_label_5.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_5 = ttk.Button(frame_5,
                                     text="Rotate", command=lambda t=tile_5: self.rotate_tile(t, image_label_5))
        rotate_button_5.grid(row=0, column=1, padx=10)

        select_button_5 = ttk.Button(frame_5, text="Select", command=lambda t=tile_5: self.select_tile(t))
        select_button_5.grid(row=0, column=2, padx=10)

        # ----------------- Tile 9 -----------------
        frame_9 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_9.pack(pady=10)

        tile_9 = Tile()
        tile_9.tile_type = TileType(9)

        image_label_9 = ttk.Label(frame_9, image=self.tile_images[9][0])
        image_label_9.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_9 = ttk.Button(frame_9,
                                     text="Rotate", command=lambda t=tile_9: self.rotate_tile(t, image_label_9))
        rotate_button_9.grid(row=0, column=1, padx=10)

        select_button_9 = ttk.Button(frame_9, text="Select", command=lambda t=tile_9: self.select_tile(t))
        select_button_9.grid(row=0, column=2, padx=10)

        # ----------------- Tile 11 -----------------
        frame_11 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_11.pack(pady=10)

        tile_11 = Tile()
        tile_11.tile_type = TileType(11)

        image_label_11 = ttk.Label(frame_11, image=self.tile_images[11][0])
        image_label_11.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_11 = ttk.Button(frame_11,
                                      text="Rotate", command=lambda t=tile_11: self.rotate_tile(t, image_label_11))
        rotate_button_11.grid(row=0, column=1, padx=10)

        select_button_11 = ttk.Button(frame_11, text="Select", command=lambda t=tile_11: self.select_tile(t))
        select_button_11.grid(row=0, column=2, padx=10)

        # ----------------- Tile 13 -----------------
        frame_13 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_13.pack(pady=10)

        tile_13 = Tile()
        tile_13.tile_type = TileType(13)

        image_label_13 = ttk.Label(frame_13, image=self.tile_images[13][0])
        image_label_13.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_13 = ttk.Button(frame_13,
                                      text="Rotate", command=lambda t=tile_13: self.rotate_tile(t, image_label_13))
        rotate_button_13.grid(row=0, column=1, padx=10)

        select_button_13 = ttk.Button(frame_13, text="Select", command=lambda t=tile_13: self.select_tile(t))
        select_button_13.grid(row=0, column=2, padx=10)

        # ----------------- Tile 17 -----------------
        frame_17 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_17.pack(pady=10)

        tile_17 = Tile()
        tile_17.tile_type = TileType(17)

        image_label_17 = ttk.Label(frame_17, image=self.tile_images[17][0])
        image_label_17.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_17 = ttk.Button(frame_17,
                                      text="Rotate", command=lambda t=tile_17: self.rotate_tile(t, image_label_17))
        rotate_button_17.grid(row=0, column=1, padx=10)

        select_button_17 = ttk.Button(frame_17, text="Select", command=lambda t=tile_17: self.select_tile(t))
        select_button_17.grid(row=0, column=2, padx=10)

        # ----------------- Tile 21 -----------------
        frame_21 = ttk.Frame(self.frame, borderwidth=2, relief="solid")
        frame_21.pack(pady=10)

        tile_21 = Tile()
        tile_21.tile_type = TileType(21)

        image_label_21 = ttk.Label(frame_21, image=self.tile_images[21][0])
        image_label_21.grid(row=0, column=0, rowspan=2, padx=5)

        rotate_button_21 = ttk.Button(frame_21,
                                      text="Rotate", command=lambda t=tile_21: self.rotate_tile(t, image_label_21))
        rotate_button_21.grid(row=0, column=1, padx=10)

        select_button_21 = ttk.Button(frame_21, text="Select", command=lambda t=tile_21: self.select_tile(t))
        select_button_21.grid(row=0, column=2, padx=10)

    def rotate_tile(self, tile, image_label: ttk.Label):
        tile.rotate_tile()
        image_label.config(image=self.tile_images[tile.tile_type.value][0])
        image_label.image = self.tile_images[tile.tile_type.value][0]
        print("Tile rotated: ", tile.tile_type)

    def select_tile(self, tile):
        self.main_gui.selected_tile = tile
        print("Tile selected: ", tile.tile_type)

    def build(self):
        print("Build button clicked")

        try:
            self.main_gui.transformed_table = self.table_obj.transform_table()

            # make all other tiles EMPTY if it is not in transform_table
            flat_transformed_table = self.table_obj.flat_nested_list(self.main_gui.transformed_table)
            locs_in_transformed_table = [elem[0] for elem in flat_transformed_table]

            # iterate over the table_obj
            for i in range(self.table_obj.table_x):
                for j in range(self.table_obj.table_y):
                    if (i, j) not in locs_in_transformed_table:
                        self.main_gui.table_section.remove_tile(i, j)

            # open a popup window to enter the circuit details
            self.main_gui.open_build_popup()

        except Exception as e:
            print("Error: ", e)
            self.main_gui.show_error_popup("Error")
