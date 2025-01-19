import json
import os
import subprocess
import sys
import time

from PyQt6.QtCore import QProcess, Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QTextEdit, QProgressDialog, QComboBox, QTextBrowser, \
    QVBoxLayout, QHBoxLayout  # Add this import
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QCheckBox, QPushButton, \
    QSpinBox, QMessageBox, QApplication

import nba_gpp_simulator
import nba_swap_sims
from nba_optimizer import NBA_Optimizer

# In your MainApp class, add these imports at the top:
from PyQt6.QtCore import QThread, pyqtSignal

from utils import resource_path  # Local import from same directory



def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)
    print("\nCurrent working directory resource:", os.getcwd())
    base_path = project_root
    return_path = os.path.join(base_path, relative_path)
    print("Current resource path:", return_path)

    # try:
    #     # PyInstaller creates a temp folder and stores path in _MEIPASS
    #     base_path = sys._MEIPASS
    # except Exception:
    #     base_path = os.path.abspath(os.path.dirname(__file__))
    return return_path

def get_output_path(filename):
    """Get the correct path for output files."""
    output_dir = resource_path('dk_output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, filename)


def print_debug_info():
    import os
    import sys
    print("=== Debug Info ===")
    #print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Parent directory: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    if getattr(sys, 'frozen', False):
        print(f"Running as executable")
        print(f"Executable path: {sys.executable}")
        print(f"Executable directory: {os.path.dirname(sys.executable)}")
        print(f"_MEIPASS: {getattr(sys, '_MEIPASS', 'Not found')}")
    print(f"Directory contents: {os.listdir(os.path.dirname(sys.executable))}")
    print("=================")

class SwapSimThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, num_iterations, site, num_uniques, num_lineups, min_salary, projection_minimum, csv_path):
        super().__init__()
        self.num_iterations = num_iterations
        self.site = site
        self.num_uniques = num_uniques
        self.num_lineups = num_lineups
        self.min_salary = min_salary
        self.projection_minimum = projection_minimum
        self.csv_path = csv_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def run(self):
        try:
            # Create simulation instance
            sim_to = nba_swap_sims.NBA_Swaptimizer_Sims(
                num_iterations=self.num_iterations,
                site=self.site,
                num_uniques=self.num_uniques,
                num_lineup_sets=self.num_lineups,
                min_salary=self.min_salary,
                projection_minimum=self.projection_minimum,
                contest_path=self.csv_path,
                is_subprocess=True
            )

            # Override print function
            def progress_print(*args):
                message = ' '.join(map(str, args))
                self.progress.emit(message)
                print(message)  # Keep console output

            # Replace the print function in your simulation instance
            start_time = time.time()
            sim_to.print = progress_print

            # Run simulation steps
            self.progress.emit("Starting swaptimization process...")
            sim_to.swaptimize()

            self.progress.emit("Computing best guesses...")
            sim_to.compute_best_guesses_parallel()

            self.progress.emit("Running tournament simulation...")
            sim_to.run_tournament_simulation()

            self.progress.emit("Generating output...")
            sim_to.output()
            self.progress.emit(f"Completed lineup swap simulation in {time.time() - start_time:.1f} seconds")

            self.finished.emit(True, "Simulation completed successfully")


        except Exception as e:
            error_message = f"Error in swap simulation: {str(e)}"
            self.progress.emit(error_message)
            self.finished.emit(False, error_message)

class SimulationThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, sim_params):
        super().__init__()
        self.sim_params = sim_params



    def run(self):
        try:
            # Unpack parameters
            site, field_size, num_iterations, use_contest_data, use_file_upload, min_salary, projection_minimum, num_lineups, csv_path = self.sim_params

            sim = nba_gpp_simulator.NBA_GPP_Simulator(
                site,
                field_size,
                num_iterations,
                use_contest_data,
                use_file_upload,
                min_salary,
                projection_minimum,
                num_lineups,
                csv_path,

            )


            # Override print function
            def progress_print(*args):
                message = ' '.join(map(str, args))
                self.progress.emit(message)
                print(message)  # Keep console output

            sim.print = progress_print

            self.progress.emit("Generating field lineups...")
            sim.generate_field_lineups()

            self.progress.emit("Running tournament simulation...")
            sim.run_tournament_simulation()

            self.progress.emit("Outputting results...")
            sim.output()

            self.finished.emit(True, "Tournament simulation completed!")
        except Exception as e:
            self.finished.emit(False, str(e))

class NbaSimsMainMenu(QMainWindow):


    def __init__(self):
        super().__init__()
        print_debug_info()  # Add this here
        self.full_filenames = {}  # Dictionary to store {display_name: full_filename}

        self.executor = None
        self.setWindowTitle("NBA DFS Tool")

        # Combine all styles into a single setStyleSheet call
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: #2E3440;  /* Dark background */
                        color: white;               /* Default text color */
                    }

                    QPushButton {
                        background-color: #5E81AC;
                        color: white;
                        font-size: 16px;
                        font-weight: regular;
                        border-radius: 10px;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #81A1C1;
                    }

                    QLineEdit {
                        background-color: #3B4252;  /* Dark background */
                        color: white;               /* White text */
                        border: 1px solid #555555;  /* Border color */
                        border-radius: 5px;         /* Rounded corners */
                        padding: 5px;               /* Inner padding */
                        font-size: 16px;            /* Font size */
                        font-family: Arial;         /* Font name */
                        font-weight: regular;          /* Font weight (bold, normal, etc.) */
                    }

                    QLineEdit:focus {
                        border: 1px solid #88C0D0;  /* Highlighted border on focus */
                        background-color: #434C5E;  /* Slightly lighter background */
                    }

                    QComboBox {
                        background-color: #3B4252;  /* Dark background */
                        color: white;               /* White text */
                        border: 1px solid #555555;  /* Border color */
                        border-radius: 5px;         /* Rounded corners */
                        padding: 5px;               /* Inner padding */
                        font-size: 14px;            /* Font size */
                    }

                      QSpinBox {
                        width: 64px;
                        background-color: #3B4252;  /* Dark background */
                        color: white;               /* White text */
                        border: 1px solid #555555;  /* Border color */
                        border-radius: 5px;         /* Rounded corners */
                        padding: 5px;               /* Inner padding */
                        font-size: 14px;            /* Font size */
                        font-weight: regular;          /* Font weight (bold, normal, etc.) */
                    }

                    QSpinBox::up-button, QSpinBox::down-button {
                        width: 32px;  /* Wider button area for arrows */
                        height: 16px; /* Taller button area for arrows */
                    }

                     QSpinBox::up-arrow {
                        width: 32px;  /* Increase arrow width */
                        height: 16px; /* Increase arrow height */
                    }

                    QSpinBox::down-arrow {
                        width: 32px;  /* Increase arrow width */
                        height: 16px; /* Increase arrow height */
                    }

                    QComboBox::drop-down {
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 40px;
                        border-left: 1px solid #555555;
                        background-color: #2E3440;  /* Dropdown arrow background */
                    }

                    QComboBox QAbstractItemView {
                        background-color: #3B4252;  /* Dropdown list background */
                        color: white;               /* Text color */
                        selection-background-color: #88C0D0; /* Selected item background */
                        selection-color: black;     /* Selected item text color */
                    }

                    QLabel {
                        color: #88C0D0;  /* Default text color */
                        font-size: 16px;
                        font-weight: regular;
                    }
                """)



        # Get the current file's directory (src)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        os.chdir(project_root)
        print("Current working directory init:", os.getcwd())


        # Get the directory of the current script and its parent
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.parent_dir = os.path.dirname(self.script_dir)
        #self.config_file = os.path.join(self.parent_dir, "config.json")
        #self.config = self.load_config(self.config_file)
        config_path = resource_path(r'config.json')
        self.config = self.load_config(config_path)


        # Default Parameters
        self.site = "dk"
        self.num_lineups = 10
        self.num_uniques = 1
        self.min_salary = int(self.config.get("min_lineup_salary", 49000))
        self.global_team_limit = int(self.config.get("global_team_limit", 4))
        self.projection_minimum = int(self.config.get("projection_minimum", 16))
        self.randomness = int(self.config.get("randomness", 100))
        self.default_var = float(self.config.get("default_var", 0.3))
        self.max_pct_off_optimal = float(self.config.get("max_pct_off_optimal", 0.3))
        self.use_contest_data = True
        self.field_size = 5000
        self.use_file_upload = False
        self.num_iterations = 5000


        # Initialize the UI
        self.init_ui()

    def load_config(self, file_path):
        """Load configuration from a JSON file."""
        try:
            config_path = resource_path(r'config.json')
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default values.")
            return {}
        except json.JSONDecodeError:
            print(f"Config file is not valid JSON. Using default values.")
            return {}

    def get_csv_files(self):
        """Get list of CSV files from contests directory."""
        try:
            csv_dir = resource_path('dk_contests')
            if os.path.exists(csv_dir):
                self.full_filenames.clear()
                files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
                for full_name in files:
                    display_name = os.path.splitext(full_name)[0]
                    self.full_filenames[display_name] = full_name
                return sorted(list(self.full_filenames.keys()), reverse=True)
        except Exception as e:
            print(f"Error loading CSV files: {e}")
        return []

    def get_selected_csv_path(self):
        """Get the full path of the selected CSV file."""
        display_name = self.csv_combo.currentText()
        if display_name:
            full_filename = self.full_filenames.get(display_name)
            if full_filename:
                return resource_path(os.path.join('dk_contests', full_filename))
        return None




    def init_ui(self):
        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        #default_button_width = 300
        #default_button_height = 50

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Initialize a grid layout instance
        layout = QGridLayout()

        # Add widgets to the grid layout
        layout.addWidget(QLabel("Number of Lineup Sets:", self), 0, 0)  # Row 0, Column 0
        self.num_lineups_input = QLineEdit(str(self.num_lineups), self)
        layout.addWidget(self.num_lineups_input, 0, 1)  # Row 0, Column 1


        layout.addWidget(QLabel("Number of Iterations:", self), 2, 0)  # Row 0, Column 0
        self.num_iterations_input = QLineEdit(str(self.num_iterations), self)
        layout.addWidget(self.num_iterations_input, 2, 1)  # Row 0, Column 1

        # Advanced Parameters Label
        num_repeats = 1
        repeated_string = " " * num_repeats
        layout.addWidget(QLabel(repeated_string, self), 3, 0, 1, 3)  # Row 2, spans 2 columns

        # Uniques
        layout.addWidget(QLabel("Number of Uniques:", self), 4, 0)  # Row 3, Column 0
        self.num_uniques_input = QSpinBox(self)
        self.num_uniques_input.setRange(1, 5)
        self.num_uniques_input.setSingleStep(1)
        self.num_uniques_input.setValue(1)
        layout.addWidget(self.num_uniques_input, 4, 1)  # Row 3, Column 1

        # Randomness
        layout.addWidget(QLabel("Randomness Amount:", self), 5, 0)  # Row 4, Column 0
        self.randomness_amount_input = QSpinBox(self)
        self.randomness_amount_input.setRange(0, 100)
        self.randomness_amount_input.setSingleStep(10)
        self.randomness_amount_input.setValue(self.randomness)
        layout.addWidget(self.randomness_amount_input, 5, 1)  # Row 4, Column 1

        # Min Salary
        layout.addWidget(QLabel("Minimum Salary:", self), 6, 0)  # Row 5, Column 0
        self.min_salary_input = QSpinBox(self)
        self.min_salary_input.setRange(0, 50000)
        self.min_salary_input.setSingleStep(100)
        self.min_salary_input.setValue(self.min_salary)
        layout.addWidget(self.min_salary_input, 6, 1)  # Row 5, Column 1

        # Projection Min
        layout.addWidget(QLabel("Projection Minimum:", self), 7, 0)  # Row 6, Column 0
        self.projection_minimum_input = QSpinBox(self)
        self.projection_minimum_input.setRange(0, 24)
        self.projection_minimum_input.setSingleStep(1)
        self.projection_minimum_input.setValue(self.projection_minimum)
        layout.addWidget(self.projection_minimum_input, 7, 1)  # Row 6, Column 1

        # Team Limit
        layout.addWidget(QLabel("Team Limit:", self), 8, 0)  # Row 7, Column 0
        self.global_team_limit_input = QSpinBox(self)
        self.global_team_limit_input.setRange(2, 7)
        self.global_team_limit_input.setSingleStep(1)
        self.global_team_limit_input.setValue(self.global_team_limit)
        layout.addWidget(self.global_team_limit_input, 8, 1)  # Row 7, Column 1

        # Add CSV file selector
        layout.addWidget(QLabel("Contest File:", self), 0, 2)  # Row 1, Column 0
        self.csv_combo = QComboBox(self)

        # Populate combo box with CSV files
        csv_files = self.get_csv_files()
        self.csv_combo.addItems(csv_files)

        # Apply consistent styling
        self.csv_combo.setStyleSheet("""
            QComboBox {
                background-color: #3B4252;
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        layout.addWidget(self.csv_combo, 1, 2)  # Row 1, Column 1

        # Action Buttons
        # Update Parameters Button
        btn_update_params = QPushButton("Update Parameters", self)
        #btn_update_params.setFixedSize(default_button_width, 50)
        btn_update_params.clicked.connect(self.update_parameters)
        layout.addWidget(btn_update_params, 9, 0, 1, 3)  # Row 1, spans 2 columns

        btn_opto = QPushButton("Optimize Lineups", self)
        #btn_opto.setFixedSize(default_button_width, default_button_height)
        btn_opto.clicked.connect(self.run_opto)
        layout.addWidget(btn_opto, 10, 0, 1, 3)  # Row 8, spans 2 columns

        btn_sim = QPushButton("Run Tournament Simulation", self)
        #btn_sim.setFixedSize(default_button_width, default_button_height)
        btn_sim.clicked.connect(self.run_sim)
        layout.addWidget(btn_sim, 11, 0, 1, 3)  # Row 9, spans 2 columns

        btn_swap_sim = QPushButton("Late Swap Simulation", self)
        #btn_swap_sim.setFixedSize(default_button_width, default_button_height)
        btn_swap_sim.clicked.connect(self.run_swap_sim)
        layout.addWidget(btn_swap_sim, 12, 0, 1, 3)  # Row 10, spans 2 columns

        # Quit Button
        btn_quit = QPushButton("Quit", self)
        #btn_quit.setFixedSize(default_button_width, default_button_height)
        btn_quit.clicked.connect(self.close)
        layout.addWidget(btn_quit, 13, 0, 1, 3)  # Row 11, spans 2 columns

        # Add progress display
        self.progress_display = QTextEdit(self)
        self.progress_display.setReadOnly(True)
        layout.addWidget(self.progress_display, 14, 0, 1, 3)  # Add below quit button

        central_widget.setLayout(layout)

    def update_parameters(self):
        # Retrieve the current values from the input fields
        try:
            self.site = "dk"
            self.num_lineups = int(self.num_lineups_input.text())
            self.num_uniques = int(self.num_uniques_input.text())
            #self.use_contest_data = self.use_contest_data_checkbox.isChecked()
            self.use_contest_data = True
            #self.field_size = int(self.field_size_input.text())
            self.field_size = 5000
            #self.use_file_upload = self.use_file_upload_checkbox.isChecked()
            self.num_iterations = int(self.num_iterations_input.text())
            self.randomness = self.randomness_amount_input.value()
            self.min_salary = self.min_salary_input.value()
            self.projection_minimum = self.projection_minimum_input.value()
            self.global_team_limit = self.global_team_limit_input.value()

            #QMessageBox.information(self, "Success", "Parameters updated successfully!")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input: {e}")

    def run_opto(self):
        try:
            self.update_parameters()
            opto = NBA_Optimizer(self.site, self.num_lineups, self.num_uniques, self.min_salary, self.randomness,
                                 self.projection_minimum, self.global_team_limit)
            opto.optimize()
            opto.output()
            QMessageBox.information(self, "Success", "Lineup optimization completed!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def run_sim(self):
        try:
            self.update_parameters()

            # Get the selected CSV file path
            csv_path = self.get_selected_csv_path()
            if not csv_path:
                QMessageBox.warning(self, "Warning", "Please select a CSV file first!")
                return

            # Create progress dialog with fixed size
            self.progress_dialog = QProgressDialog("Running simulation...", None, 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setCancelButton(None)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.progress_dialog.setWindowTitle('Running')

            # Set fixed size for the dialog
            self.progress_dialog.setFixedWidth(400)  # Adjust width as needed
            self.progress_dialog.setMinimumHeight(100)  # Minimum height

            # Make the label inside the dialog wrap text
            label = self.progress_dialog.findChild(QLabel)
            if label:
                label.setWordWrap(True)
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


            # Prepare simulation parameters
            sim_params = (self.site, self.field_size, self.num_iterations,
                          self.use_contest_data, self.use_file_upload,
                          self.min_salary, self.projection_minimum,
                          self.num_lineups,
                          csv_path,
                          )

            # Create and setup simulation thread
            self.sim_thread = SimulationThread(sim_params)
            self.sim_thread.progress.connect(self.update_progress)
            self.sim_thread.finished.connect(self.simulation_finished)

            # Start simulation
            self.sim_thread.start()
            self.progress_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def update_progress(self, message):
        try:
            # Update progress text
            self.progress_dialog.setLabelText(message)

            # Process events to keep GUI responsive
            QApplication.processEvents()

        except Exception as e:
            print(f"Error updating progress: {e}")


    def simulation_finished(self, success, message):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", f"An error occurred: {message}")


    def run_swap_sim(self):
        try:
            self.update_parameters()

            # Get the selected CSV file path
            csv_path = self.get_selected_csv_path()
            if not csv_path:
                QMessageBox.warning(self, "Warning", "Please select a CSV file first!")
                return
            # Create progress dialog with a text browser for better output display
            self.progress_dialog = QProgressDialog(self)
            self.progress_dialog.setWindowTitle("Swap Simulation")
            self.progress_dialog.setMinimum(0)
            self.progress_dialog.setMaximum(0)
            self.progress_dialog.setCancelButton(None)
            self.progress_dialog.setMinimumWidth(800)
            self.progress_dialog.setMinimumHeight(400)

            # Add a close button with fixed width
            self.close_button = QPushButton("Close")
            self.close_button.clicked.connect(self.progress_dialog.close)
            self.close_button.setFixedWidth(200)
            self.close_button.hide()  # Hide initially

            # Create text browser for output with adjusted height
            self.output_browser = QTextBrowser()
            self.output_browser.setMinimumWidth(780)
            self.output_browser.setMinimumHeight(340)  # Reduced height to leave room for button
            self.output_browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Ensure scrollbar shows when needed

            # Create layout for progress dialog with spacing
            layout = QVBoxLayout()
            layout.addWidget(self.output_browser)

            # Create a horizontal layout for the button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(self.close_button)
            button_layout.addStretch()

            # Add button layout to main layout with spacing
            layout.addLayout(button_layout)
            layout.setSpacing(10)  # Add spacing between elements
            layout.setContentsMargins(10, 10, 10, 10)  # Add margins around the layout

            # Get the progress dialog's content widget
            content = self.progress_dialog.findChild(QWidget)
            if content:
                content.setLayout(layout)

            # Create swap simulation thread
            self.swap_thread = SwapSimThread(
                self.num_iterations,
                self.site,
                self.num_uniques,
                self.num_lineups,
                self.min_salary,
                self.projection_minimum,
                csv_path
            )

            # Connect signals
            self.swap_thread.progress.connect(self.update_swap_progress)
            self.swap_thread.finished.connect(self.swap_sim_finished)

            # Start simulation
            self.swap_thread.start()
            self.progress_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def update_swap_progress(self, message):
        # Print to console
        print(message)

        # Update GUI
        if hasattr(self, 'output_browser') and self.output_browser is not None:
            self.output_browser.append(message)
            scrollbar = self.output_browser.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            QApplication.processEvents()


    def swap_sim_finished(self, success, message):
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            # Show the close button instead of auto-closing
            if hasattr(self, 'close_button'):
                self.close_button.show()

            # Update dialog title to show completion
            self.progress_dialog.setWindowTitle("Swap Simulation - Completed")

        if success:
            completion_message = "Simulation completed successfully!"
            print(completion_message)  # Print to console
            # Show success message but don't close progress dialog
            QMessageBox.information(self, "Success", completion_message)
        else:
            error_message = f"Swap simulation failed: {message}"
            print(error_message)  # Print to console
            QMessageBox.critical(self, "Error", error_message)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Necessary for Windows
    app = QApplication(sys.argv)
    main_window = NbaSimsMainMenu()
    main_window.show()
    sys.exit(app.exec())

















