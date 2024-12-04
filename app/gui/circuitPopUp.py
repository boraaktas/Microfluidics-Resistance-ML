import io
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

from app.utils import Constants
from src.maze_functions import plot_other_components
from src.modelling_3D import build_whole_circuit


class CircuitPopup:
    def __init__(self, root, ALL_GENERATED_COMPONENTS, table_obj):
        self.root = root
        self.ALL_GENERATED_COMPONENTS = ALL_GENERATED_COMPONENTS
        self.table_obj = table_obj

        self.toggle_base_checkbox_var = tk.BooleanVar()

        # Create images and models
        (self.images_2d,
         self.model_3d_without_base,
         self.model_3d_with_base,
         self.updated_dict,
         self.most_upper_row,
         self.most_left_col) = self.create_images_and_models()

    def create_images_and_models(self, image_size=(90, 90)):
        most_upper_row = min([cell_loc[0] for cell_loc in self.ALL_GENERATED_COMPONENTS.keys()])
        most_left_col = min([cell_loc[1] for cell_loc in self.ALL_GENERATED_COMPONENTS.keys()])

        # Update the dict according to the most upper row, lowest row, most left col, most right col
        updated_dict = {}
        for cell_loc in self.ALL_GENERATED_COMPONENTS.keys():
            row, col = cell_loc
            updated_dict[(row - most_upper_row, col - most_left_col)] = self.ALL_GENERATED_COMPONENTS[cell_loc]

        dict_for_3d_model = {}
        max_x = max([cell_loc[0] for cell_loc in updated_dict.keys()])
        for key in updated_dict.keys():
            new_key = (key[1], max_x - key[0])
            dict_for_3d_model[new_key] = updated_dict[key]

        updated_most_lower_row = max([cell_loc[0] for cell_loc in updated_dict.keys()])
        updated_most_right_col = max([cell_loc[1] for cell_loc in updated_dict.keys()])

        # Create 2D images
        images = {}
        for i in range(updated_most_lower_row + 1):
            for j in range(updated_most_right_col + 1):
                empty_cell_plot = plot_other_components("EMPTY", show_plot=False, shape=(20, 20))
                resized_empty_cell_plot = self.resize_figure(empty_cell_plot, image_size)
                img = ImageTk.PhotoImage(resized_empty_cell_plot)
                images[(i, j)] = img

        for cell_loc in updated_dict.keys():
            (cell_res_matrix, cell_fig,
             pipe_width, pipe_height, pipe_fillet_radius,
             cell_type, cell_coming_dir) = updated_dict[cell_loc]
            row, col = cell_loc

            resized_image = self.resize_figure(cell_fig, image_size)

            img = ImageTk.PhotoImage(resized_image)
            images[(row, col)] = img

        # Create 3D models
        combined_model, combined_model_with_base = build_whole_circuit(dict_for_3d_model, show_model=False)

        return images, combined_model, combined_model_with_base, updated_dict, most_upper_row, most_left_col

    @staticmethod
    def resize_figure(fig, size):
        """Resize a Matplotlib figure to a PIL image with the given size."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize(size, Image.LANCZOS)
        return img

    def open_circuit_popup(self):

        popup = tk.Toplevel()
        popup.title("Generated Circuit")
        popup.resizable(True, True)

        # Make mainGUI unresponsive
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)

        # Open in the center of the screen and make it on top of the mainGUI
        popup.update_idletasks()
        popup_width = int(self.root.winfo_screenwidth() * 0.9)
        popup_height = int(self.root.winfo_screenheight() * 0.9)

        x_coordinate = int((self.root.winfo_x() + self.root.winfo_width() / 2) - (popup_width / 2))
        y_coordinate = int((self.root.winfo_y() + self.root.winfo_height() / 2) - (popup_height / 2))

        popup.geometry(f"{popup_width}x{popup_height}+{x_coordinate}+{y_coordinate}")

        # Create a main frame to hold the scrollable content and the panel
        main_frame = ttk.Frame(popup)
        main_frame.pack(fill='both', expand=True)

        container_2d = ttk.Frame(main_frame)
        container_2d.pack(side="left", fill="both", expand=True)

        # Create a canvas and add scrollbars to it
        canvas = tk.Canvas(container_2d)
        scrollbar_y_2d = ttk.Scrollbar(container_2d, orient="vertical", command=canvas.yview)
        scrollbar_x_2d = ttk.Scrollbar(container_2d, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y_2d.set, xscrollcommand=scrollbar_x_2d.set)

        scrollbar_y_2d.pack(side="right", fill="y")
        scrollbar_x_2d.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Create the grid with images in the scrollable content area
        for cell_loc, img in self.images_2d.items():
            img_label = ttk.Label(scrollable_frame, image=img)
            img_label.image = img
            img_label.grid(row=cell_loc[0], column=cell_loc[1], padx=0, pady=0)

        # Create a panel frame to hold both the scrollable area and the button below it
        panel_frame = ttk.Frame(main_frame, width=int(popup_width * 0.3))
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

        title_selected_groups_font = ("Arial", 16, "bold")
        title_selected_groups_label = ttk.Label(scrollable_panel_frame,
                                                text="Selected Pipe Features",
                                                font=title_selected_groups_font)
        title_selected_groups_label.grid(row=0, column=0, columnspan=2, pady=(5, 5), sticky="n")
        # if it is a START, display the pipe features
        row_index = 1
        for i, cell_loc in enumerate(self.updated_dict.keys()):
            table_loc = (cell_loc[0] + self.most_upper_row, cell_loc[1] + self.most_left_col)

            if self.table_obj.table[table_loc[0]][table_loc[1]].tile_type in Constants.STARTER_TYPES:
                cell_res_matrix, cell_fig, pipe_width, pipe_height, pipe_fillet_radius, cell_type, cell_coming_dir = \
                    self.updated_dict[cell_loc]

                # Display the cell location
                ttk.Label(scrollable_panel_frame, text=f"Circuit Pipe That Starts at Cell"
                                                       f" {cell_loc}:").grid(row=row_index,
                                                                             column=0,
                                                                             columnspan=2,
                                                                             sticky="w",
                                                                             padx=10)
                row_index += 1

                # Display the pipe features of the circuit: width, height, fillet radius
                ttk.Label(scrollable_panel_frame, text=f"PipeWidth: {pipe_width:.2f} mm").grid(row=row_index,
                                                                                        column=0,
                                                                                        columnspan=2,
                                                                                        sticky="w",
                                                                                        padx=10)
                row_index += 1
                ttk.Label(scrollable_panel_frame, text=f"Pipe Height: {pipe_height:.2f} mm").grid(row=row_index,
                                                                                               column=0,
                                                                                               columnspan=2,
                                                                                               sticky="w",
                                                                                               padx=10)
                row_index += 1
                ttk.Label(scrollable_panel_frame, text=f"Fillet Radius: {pipe_fillet_radius:.2f} mm").grid(row=row_index,
                                                                                                       column=0,
                                                                                                       columnspan=2,
                                                                                                       sticky="w",
                                                                                                       padx=10)
                row_index += 1

                ttk.Separator(scrollable_panel_frame, orient='horizontal').grid(row=row_index, column=0, columnspan=2,
                                                                                sticky="ew", pady=5)
                row_index += 1

        # Centered and styled Flow Rates title with no extra padding
        title_font = ("Arial", 16, "bold")
        title_label = ttk.Label(scrollable_panel_frame, text="Flow Rates", font=title_font)
        title_label.grid(row=row_index, column=0, columnspan=2, pady=(5, 5), sticky="n")

        row_index += 1
        for i, cell_loc in enumerate(self.updated_dict.keys()):
            table_loc = (cell_loc[0] + self.most_upper_row, cell_loc[1] + self.most_left_col)

            if self.table_obj.table[table_loc[0]][table_loc[1]].tile_type in Constants.Q_TILES \
                    or self.table_obj.table[table_loc[0]][table_loc[1]].tile_type in Constants.END_TYPES:
                desired_flow_rate = self.table_obj.table[table_loc[0]][table_loc[1]].flow_rate_in_this_cell
                generated_flow_rate = self.table_obj.table[table_loc[0]][table_loc[1]].generated_flow_rate_in_this_cell

                # Display the cell location
                ttk.Label(scrollable_panel_frame, text=f"Cell {cell_loc}:").grid(row=row_index, column=0, columnspan=2,
                                                                                 sticky="w", padx=10)
                row_index += 1

                # Display Desired Q and Generated Q side by side with 2 decimal points
                ttk.Label(scrollable_panel_frame, text=f"Desired Q: {desired_flow_rate:.2f}").grid(row=row_index,
                                                                                                   column=0,
                                                                                                   sticky="w",
                                                                                                   padx=10)
                ttk.Label(scrollable_panel_frame, text=f"Generated Q: {generated_flow_rate:.2f}").grid(row=row_index,
                                                                                                       column=1,
                                                                                                       sticky="w",
                                                                                                       padx=10)
                row_index += 1

                ttk.Separator(scrollable_panel_frame, orient='horizontal').grid(row=row_index, column=0, columnspan=2,
                                                                                sticky="ew", pady=5)
                row_index += 1

        # Update the scroll region after adding widgets
        scrollable_panel_frame.update_idletasks()
        panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))

        # Add buttons to the panel
        download_button = ttk.Button(panel_frame, text="Download 3D Model", command=self.download_3d_model)
        download_button.pack(side="bottom", pady=10, fill='x')

        base_checkbox = ttk.Checkbutton(panel_frame, text="Add Base", variable=self.toggle_base_checkbox_var)
        base_checkbox.pack(side="bottom", pady=10)

    def download_3d_model(self):
        selected_model = self.model_3d_with_base if self.toggle_base_checkbox_var.get() else self.model_3d_without_base

        # Select the file path to save the 3D model
        file_path = filedialog.asksaveasfilename(defaultextension=".stl",
                                                 filetypes=[("STL files", "*.stl"), ("All files", "*.*")])

        if file_path:
            selected_model.export(file_path)
