import io
import tkinter as tk
import traceback

import numpy as np
import ttkbootstrap as ttk
from PIL import ImageTk, Image
from matplotlib import pyplot as plt

from app.utils import (load_images, load_res_bounds, load_prediction_model,
                       R_calculator, Q_calculator, ResistanceGenerator, Constants, Table, Tile, TileType)
from src.maze_functions import plot_other_components
from src.modelling_3D import build_whole_circuit
from .menuGUI import Menu_Section
from .tableGUI import Table_Section


class Main_Section:
    def __init__(self, root, length=20, width=20):
        self.root = root
        self.root.title("Microfluidics Maze Builder")
        # fix the size of the window
        self.root.resizable(False, False)

        # open it in the center of the screen
        window_width = 1000
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_coordinate = int((screen_width / 2) - (window_width / 2))
        y_coordinate = int((screen_height / 2) - (window_height / 2))
        self.root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_coordinate, y_coordinate))

        self.images = load_images()
        self.res_bounds = load_res_bounds()
        self.prediction_model = load_prediction_model()

        self.table_length = length
        self.table_width = width
        self.table_obj = Table(self.table_length, self.table_width)

        self.selected_tile = Tile()

        self.entries = []  # Add a reference to the entries in the popup window
        self.popup = None  # Add a reference to the popup window

        self.transformed_table = None
        self.resistance_dict = None

        Menu_Section(self)
        Table_Section(self)

    def show_error_popup(self, message):
        popup = tk.Toplevel()
        popup.title("Error")
        popup.resizable(False, False)

        # make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # open in the center of the screen and
        # make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = 350
        popup_height = 80

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry("{}x{}+{}+{}".format(popup_width, popup_height, x_coordinate, y_coordinate))

        label = ttk.Label(popup, text=message.split("\n")[0], anchor="center")
        label.pack(pady=10)

        label = ttk.Label(popup, text=message.split("\n")[1], anchor="center")
        label.pack(pady=1)

    def open_build_popup(self):
        entry_pressures = self.table_obj.find_entry_pressures()
        flow_rate_calculators = self.table_obj.find_flow_rate_calculators()
        exit_pressures = self.table_obj.find_exit_pressures()

        if not entry_pressures:
            self.show_error_popup("No entry pressures found")
            return

        if not flow_rate_calculators:
            self.show_error_popup("No flow rate calculators found")
            return

        if not exit_pressures:
            self.show_error_popup("No exit pressures found")
            return

        popup = tk.Toplevel()
        popup.title("Enter Inputs")
        popup.resizable(False, False)

        # make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # open in the center of the screen and make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = 300
        popup_height = max(300, min(45 * (len(entry_pressures) + len(flow_rate_calculators) + 1), 600))

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry("{}x{}+{}+{}".format(popup_width, popup_height, x_coordinate, y_coordinate))

        # Create a frame inside the popup to hold the scrollable content
        container = ttk.Frame(popup)
        container.pack(fill='both', expand=True)

        # Create a canvas and add scrollbars to it
        canvas = tk.Canvas(container)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_y.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Add user input fields for each entry pressure and flow rate calculator
        row = 0
        self.entries = []  # Clear previous entries

        for entry_pressure in entry_pressures:
            label = ttk.Label(scrollable_frame, text=f"Entry Pressure ({entry_pressure[0][0] + 1},"
                                                     f" {entry_pressure[0][1] + 1}):", anchor="center")
            label.grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(scrollable_frame, width=5)
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.entries.append(((entry_pressure[0][0], entry_pressure[0][1]), entry))
            row += 1

        for flow_rate_calculator in flow_rate_calculators:
            label = ttk.Label(scrollable_frame, text=f"Desired Flow Rate ({flow_rate_calculator[0][0] + 1},"
                                                     f" {flow_rate_calculator[0][1] + 1}):", anchor="center")
            label.grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(scrollable_frame, width=5)
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.entries.append(((flow_rate_calculator[0][0], flow_rate_calculator[0][1]), entry))
            row += 1

        # Add a button to submit the inputs
        submit_button = ttk.Button(scrollable_frame, text="Submit", command=self.submit_inputs, width=20)
        submit_button.grid(row=row, column=0, columnspan=2, pady=10)

        self.popup = popup

    def submit_inputs(self):
        inputs = {}
        for (position, entry) in self.entries:
            value = entry.get()
            try:
                value = float(value)
                if value <= 0:
                    self.table_obj.make_all_tiles_attributes_none()
                    raise ValueError("All fields must be filled with positive numbers.")
            except ValueError:
                self.table_obj.make_all_tiles_attributes_none()
                self.show_error_popup("All fields must be filled with numbers.")

            type_in_position = self.table_obj.table[position[0]][position[1]].tile_type
            if type_in_position in Constants.STARTER_TYPES:
                inputs[position] = {"P": value, "I": None}
            elif type_in_position in Constants.Q_TILES:
                inputs[position] = {"P": None, "I": value}
            else:
                self.table_obj.make_all_tiles_attributes_none()
                raise ValueError("Invalid tile type.")

        exit_pressures = self.table_obj.find_exit_pressures()
        for i in range(len(exit_pressures)):
            inputs[exit_pressures[i][0]] = {"P": 0, "I": None}

        print(inputs)

        # go all tiles in inputs and update the transformed table
        for key in inputs.keys():
            tile_obj = self.table_obj.table[key[0]][key[1]]
            if inputs[key]["P"] is not None:
                tile_obj.pressure_in_this_cell = inputs[key]["P"]
            if inputs[key]["I"] is not None:
                tile_obj.flow_rate_in_this_cell = inputs[key]["I"]

        success_flow_rates = False
        try:
            flow_rate_dict = Q_calculator(self.transformed_table)
            for key in flow_rate_dict.keys():
                tile_obj = self.table_obj.table[key[0]][key[1]]
                tile_obj.flow_rate_in_this_cell = flow_rate_dict[key]

            self.table_obj.set_flow_rates_for_other_components(self.transformed_table)

            success_flow_rates = True
        except Exception as e:
            self.table_obj.make_all_tiles_attributes_none()
            print("Error: ", e)
            # print the stack trace
            traceback.print_exc()
            self.show_error_popup("Conservation of mass is not satisfied.\n" + str(e))

        success_resistances = False
        if success_flow_rates:
            try:
                self.resistance_dict = R_calculator(self.transformed_table, self.res_bounds)
                for key in self.resistance_dict.keys():
                    print(f"{key}: {self.resistance_dict[key]}")

                # it will update transformed_table with the selected combinations
                self.table_obj.set_selected_comb_to_all_components(self.transformed_table)

                # update the resistance_in_this_cell attribute of the tiles
                for i in range(len(self.resistance_dict)):
                    # get the key and value of the resistance_dict
                    current_key = list(self.resistance_dict.keys())[i]
                    key_resistance = current_key[0]
                    for value in self.resistance_dict[current_key]:
                        coords = value[0]
                        self.table_obj.table[coords[0]][coords[1]].resistance_in_this_cell = key_resistance

                success_resistances = True
                print("\n\n\n")
            except Exception as e:
                self.table_obj.make_all_tiles_attributes_none()
                print("Error: ", e)
                # print the stack trace
                traceback.print_exc()
                self.show_error_popup("Infeasible circuit.\n" + "Each cell resistance should be between 1 and 70.")

        if success_resistances:
            # close this popup
            if self.popup is not None:
                self.popup.destroy()
            # open a new window to play an image that represents running algorithms
            self.open_algorithm_popup()

    def open_algorithm_popup(self):
        popup = tk.Toplevel()
        popup.title("Generating Mazes")
        popup.resizable(False, False)

        # make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # open in the center of the screen and make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = 300
        popup_height = 100

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry("{}x{}+{}+{}".format(popup_width, popup_height, x_coordinate, y_coordinate))

        label = ttk.Label(popup, text="Generating mazes...", anchor="center")
        label.pack(pady=10)

        # write percentage that is generated in the label
        total_cell_count = sum([len(self.resistance_dict[key]) for key in self.resistance_dict.keys()])
        # Initialize generated cell count
        generated_cell_count = tk.IntVar(value=0)
        # write the label to show the progress put a progress bar
        progress_bar = ttk.Progressbar(popup, maximum=total_cell_count, variable=generated_cell_count, length=300)
        progress_bar.pack(pady=10)

        RG = ResistanceGenerator(self.prediction_model)

        ALL_GENERATED_COMPONENTS \
            : dict[tuple[int, int], tuple[np.ndarray, plt.Figure, float, float, float, TileType, str]] = {}

        entry_pressures = self.table_obj.find_entry_pressures()
        flow_rate_calculators = self.table_obj.find_flow_rate_calculators()
        exit_pressures = self.table_obj.find_exit_pressures()
        divisions = self.table_obj.find_divisions()

        all_other_components = entry_pressures + flow_rate_calculators + exit_pressures + divisions

        # add inlet, flow rate calculator, exit, and division components to the generated components
        for component in all_other_components:
            cur_component_loc = component[0]
            cur_component_type = component[1].tile_type
            cur_component_image = plot_other_components(str(cur_component_type).split(".")[1],
                                                        show_plot=False, shape=(20, 20))

            cur_component_width = self.res_bounds[component[1].selected_comb_for_tile]['Width']
            cur_component_height = self.res_bounds[component[1].selected_comb_for_tile]['Height']
            cur_component_fillet_radius = self.res_bounds[component[1].selected_comb_for_tile]['Fillet_Radius']

            cur_component_coming_direction = component[1].coming_direction

            ALL_GENERATED_COMPONENTS[cur_component_loc] = (np.empty((21, 21)),
                                                           cur_component_image,
                                                           cur_component_width,
                                                           cur_component_height,
                                                           cur_component_fillet_radius,
                                                           cur_component_type,
                                                           cur_component_coming_direction
                                                           )

        for key in self.resistance_dict.keys():
            cell_res_value: float = key[0]
            cell_target_loc_mode: str = "east" if key[2] == 'STRAIGHT' else "north"
            pipe_width = self.res_bounds[key[1]]['Width']
            pipe_height = self.res_bounds[key[1]]['Height']
            pipe_fillet_radius = self.res_bounds[key[1]]['Fillet_Radius']
            cell_step_size_factor = 0.5
            cell_side_length = 20

            (cell_resistance_matrix,
             cell_resistance_value) = RG.generate_resistance(desired_resistance=cell_res_value,
                                                             target_loc_mode=cell_target_loc_mode,
                                                             width=pipe_width,
                                                             height=pipe_height,
                                                             fillet_radius=pipe_fillet_radius,
                                                             step_size_factor=cell_step_size_factor,
                                                             side_length=cell_side_length)

            all_cell_locs_and_types = self.resistance_dict[key]
            for cell_loc_type in all_cell_locs_and_types:
                cur_cell_loc = cell_loc_type[0]
                cur_cell_type = cell_loc_type[1].tile_type
                cur_cell_coming_direction = cell_loc_type[1].coming_direction

                cur_cell_res_matrix, cur_cell_fig = RG.get_rotated_matrix_and_figure(cell_resistance_matrix,
                                                                                     cur_cell_type)

                ALL_GENERATED_COMPONENTS[cur_cell_loc] = (cur_cell_res_matrix,
                                                          cur_cell_fig,
                                                          pipe_width,
                                                          pipe_height,
                                                          pipe_fillet_radius,
                                                          cur_cell_type,
                                                          cur_cell_coming_direction)

                # update self.transformed_table with the cell_resistance_value
                self.table_obj.table[cur_cell_loc[0]][cur_cell_loc[1]].generated_resistance_in_this_cell \
                    = cell_resistance_value

                generated_cell_count.set(generated_cell_count.get() + 1)
                popup.update()

        # when all the mazes are generated close this popup, and show new popup with all_cells_locs_and_types
        if popup is not None:
            popup.destroy()

        self.open_circuit_popup(ALL_GENERATED_COMPONENTS)

    @staticmethod
    def resize_figure(fig, size):
        """Resize a Matplotlib figure to a PIL image with the given size."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize(size, Image.LANCZOS)
        return img

    def open_circuit_popup(self, ALL_GENERATED_COMPONENTS, image_size=(90, 90)):
        most_upper_row = min([cell_loc[0] for cell_loc in ALL_GENERATED_COMPONENTS.keys()])
        most_left_col = min([cell_loc[1] for cell_loc in ALL_GENERATED_COMPONENTS.keys()])

        # Update the dict according to the most upper row, most lower row, most left col, most right col
        updated_dict = {}
        for cell_loc in ALL_GENERATED_COMPONENTS.keys():
            row, col = cell_loc
            updated_dict[(row - most_upper_row, col - most_left_col)] = ALL_GENERATED_COMPONENTS[cell_loc]

        updated_most_lower_row = max([cell_loc[0] for cell_loc in updated_dict.keys()])
        updated_most_right_col = max([cell_loc[1] for cell_loc in updated_dict.keys()])

        dict_for_3d_model = {}
        max_x = max([cell_loc[0] for cell_loc in updated_dict.keys()])
        for key in updated_dict.keys():
            new_key = (key[1], max_x - key[0])
            dict_for_3d_model[new_key] = updated_dict[key]

        popup = tk.Toplevel()
        popup.title("Generated Circuit")
        popup.resizable(False, False)

        # Make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # Open in the center of the screen and make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = 1000  # Adjusted width for the panel
        popup_height = 600

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry("{}x{}+{}+{}".format(popup_width, popup_height, x_coordinate, y_coordinate))

        # Create a main frame to hold the scrollable content and the panel
        main_frame = ttk.Frame(popup)
        main_frame.pack(fill='both', expand=True)

        # Create a frame inside the main frame to hold the scrollable content
        container = ttk.Frame(main_frame)
        container.pack(side="left", fill='both', expand=True)

        # Create a canvas and add scrollbars to it
        canvas = tk.Canvas(container)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Create a panel frame to hold both the scrollable area and the button below it
        panel_frame = ttk.Frame(main_frame, width=300)
        panel_frame.pack(side="right", fill="y")

        # Create a canvas for the scrollable part inside the panel
        panel_canvas = tk.Canvas(panel_frame)
        panel_scrollbar = ttk.Scrollbar(panel_frame, orient="vertical", command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
        panel_scrollbar.pack(side="right", fill="y")
        panel_canvas.pack(side="top", fill="both", expand=True)

        # Create a frame inside the canvas to hold the scrollable content
        scrollable_panel_frame = ttk.Frame(panel_canvas)
        panel_canvas.create_window((0, 0), window=scrollable_panel_frame, anchor="nw")

        # Configure grid to remove extra spacing
        scrollable_panel_frame.grid_columnconfigure(0, weight=1)
        scrollable_panel_frame.grid_columnconfigure(1, weight=1)

        # Centered and styled Flow Rates title with no extra padding
        title_font = ("Arial", 16, "bold")
        title_label = ttk.Label(scrollable_panel_frame, text="Flow Rates", font=title_font)
        title_label.grid(row=0, column=0, columnspan=2, pady=(5, 5), sticky="n")

        # Assuming you have a way to retrieve the desired and generated flow rates for each cell
        row_index = 1
        for i, cell_loc in enumerate(updated_dict.keys()):
            table_loc = (cell_loc[0] + most_upper_row, cell_loc[1] + most_left_col)

            if self.table_obj.table[table_loc[0]][table_loc[1]].tile_type in Constants.Q_TILES \
                    or self.table_obj.table[table_loc[0]][table_loc[1]].tile_type in Constants.END_TYPES:
                desired_flow_rate = self.table_obj.table[table_loc[0]][table_loc[1]].flow_rate_in_this_cell
                generated_flow_rate = self.table_obj.table[table_loc[0]][table_loc[1]].generated_flow_rate_in_this_cell

                # Display the cell location
                ttk.Label(scrollable_panel_frame, text=f"Cell {cell_loc}:").grid(row=row_index, column=0, columnspan=2,
                                                                                 sticky="w", padx=10)
                row_index += 1

                # Display Desired Q and Generated Q side by side
                ttk.Label(scrollable_panel_frame, text=f"Desired Q: {desired_flow_rate}").grid(row=row_index, column=0,
                                                                                               sticky="w", padx=10)
                ttk.Label(scrollable_panel_frame, text=f"Generated Q: {generated_flow_rate}").grid(row=row_index,
                                                                                                   column=1, sticky="w",
                                                                                                   padx=10)
                row_index += 1

                ttk.Separator(scrollable_panel_frame, orient='horizontal').grid(row=row_index, column=0, columnspan=2,
                                                                                sticky="ew", pady=5)
                row_index += 1

        # Update the scroll region after adding widgets
        scrollable_panel_frame.update_idletasks()
        panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))

        # Add the download button below the scrollable panel content
        download_button = ttk.Button(panel_frame, text="Download 3D Model",
                                     command=lambda: self.download_3d_model(dict_for_3d_model))
        download_button.pack(side="bottom", pady=10)

        # Create the grid with images in the scrollable content area
        for i in range(updated_most_lower_row + 1):
            for j in range(updated_most_right_col + 1):
                empty_cell_plot = plot_other_components("EMPTY", show_plot=False, shape=(20, 20))
                resized_empty_cell_plot = self.resize_figure(empty_cell_plot, image_size)
                img = ImageTk.PhotoImage(resized_empty_cell_plot)
                img_label = ttk.Label(scrollable_frame, image=img)
                img_label.image = img  # Keep a reference to avoid garbage collection
                img_label.grid(row=i, column=j, padx=0, pady=0)

        for cell_loc in updated_dict.keys():
            (cell_res_matrix, cell_fig,
             pipe_width, pipe_height, pipe_fillet_radius,
             cell_type, cell_coming_dir) = updated_dict[cell_loc]
            row, col = cell_loc  # cell_loc is a tuple (x, y)

            resized_image = self.resize_figure(cell_fig, image_size)

            img = ImageTk.PhotoImage(resized_image)
            img_label = ttk.Label(scrollable_frame, image=img)
            img_label.image = img  # Keep a reference to avoid garbage collection
            img_label.grid(row=row, column=col, padx=0, pady=0)

    def download_3d_model(self, DICT_FOR_3D_MODEL):

        for key in DICT_FOR_3D_MODEL.keys():
            print(key, DICT_FOR_3D_MODEL[key][0].shape, DICT_FOR_3D_MODEL[key][1],
                  DICT_FOR_3D_MODEL[key][2],
                  DICT_FOR_3D_MODEL[key][3], DICT_FOR_3D_MODEL[key][4],
                  DICT_FOR_3D_MODEL[key][5], DICT_FOR_3D_MODEL[key][6])

        popup = tk.Toplevel()
        popup.title("Download Options")
        popup.resizable(False, False)

        # make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # open in the center of the screen and make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = 300
        popup_height = 100

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry("{}x{}+{}+{}".format(popup_width, popup_height, x_coordinate, y_coordinate))

        label = ttk.Label(popup, text="Downloading 3D model...", anchor="center")
        label.pack(pady=10)

        # put a progress bar
        progress_bar = ttk.Progressbar(popup, maximum=100, length=300)
        progress_bar.pack(pady=10)

        combined_model, combined_model_with_base = build_whole_circuit(DICT_FOR_3D_MODEL, show_model=True)
