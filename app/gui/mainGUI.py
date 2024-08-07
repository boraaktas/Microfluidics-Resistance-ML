import tkinter as tk
import traceback

import ttkbootstrap as ttk

from .menuGUI import Menu_Section
from .tableGUI import Table_Section
from src.generative_model import GenerativeModel
from app.utils import load_images, Table, Tile, R_calculator, Q_calculator


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
            label = ttk.Label(scrollable_frame, text=f"Flow Rate Calculator ({flow_rate_calculator[0][0] + 1},"
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
            except ValueError:
                self.show_error_popup("All fields must be filled with numbers.")
            inputs[position] = value

        exit_pressures = self.table_obj.find_exit_pressures()
        for i in range(len(exit_pressures)):
            inputs[exit_pressures[i][0]] = 0

        print(inputs)

        self.transformed_table = self.table_obj.update_transformed_table(self.transformed_table,
                                                                         inputs, mode=1)
        success_flow_rates = False
        try:
            flow_rate_dict = Q_calculator(self.transformed_table)
            self.transformed_table = self.table_obj.update_transformed_table(self.transformed_table,
                                                                             flow_rate_dict, mode=2)
            success_flow_rates = True
        except Exception as e:
            print("Error: ", e)
            # print the stack trace
            traceback.print_exc()
            self.show_error_popup("Conservation of mass is not satisfied.\n" + str(e))

        success_resistances = False
        if success_flow_rates:
            try:
                self.resistance_dict = R_calculator(self.transformed_table)
                for key in self.resistance_dict.keys():
                    print(f"{key}: {self.resistance_dict[key]}")
                success_resistances = True
                print("\n\n\n")
            except Exception as e:
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

        GM = GenerativeModel(list(self.resistance_dict.keys()))
