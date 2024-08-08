import tkinter as ttk
from tkinter import Scrollbar, Canvas

from app.utils import Tile


class Table_Section:
    def __init__(self, main_gui):
        self.frame = ttk.Frame(main_gui.root, borderwidth=2, relief="solid")
        self.frame.place(relx=0, relwidth=0.7, relheight=1, anchor='nw')

        self.root = main_gui.root

        self.table_images = main_gui.images.copy()

        self.main_gui = main_gui
        self.table_length = main_gui.table_length
        self.table_width = main_gui.table_width
        self.table_obj = main_gui.table_obj

        self.buttons = []

        self.canvas = Canvas(self.frame)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollbar_y = Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x = Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_x.pack(side="bottom", fill="x")

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.create_table_ui()

    def create_table_ui(self):
        for i in range(self.table_length):
            row = []
            for j in range(self.table_width):
                button = ttk.Button(self.scrollable_frame,
                                    image=self.table_images[self.table_obj.table[i][j].tile_type.value][0],
                                    text="",
                                    bd=0)
                button.grid(row=i, column=j, padx=0, pady=0, ipadx=0, ipady=0)
                button.bind("<Button-1>", lambda event, x=i, y=j: self.change_tile(x, y))  # Left click
                button.bind("<Button-2>", lambda event, x=i, y=j: self.remove_tile(x, y))  # Right click
                row.append(button)
            self.buttons.append(row)

    def change_tile(self, x, y):
        self.table_obj.set_tile(x, y, self.main_gui.selected_tile)

        # Resize the image to cover the whole button
        image = self.table_images[self.main_gui.selected_tile.tile_type.value][0]
        self.buttons[x][y].config(image=image)

    def remove_tile(self, x, y):
        self.table_obj.set_tile(x, y, Tile())  # Set to empty tile

        # Resize the image to cover the whole button
        image = self.table_images[0][0]  # Assuming the first image is the empty tile image
        self.buttons[x][y].config(image=image)
