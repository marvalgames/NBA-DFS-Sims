import json
import os
import sys

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QGridLayout  # Add this import
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QCheckBox, QPushButton, \
    QSpinBox, QMessageBox, QApplication

import nba_gpp_simulator
from nba_optimizer import NBA_Optimizer


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.executor = None
        self.setWindowTitle("NBA DFS Tool")
        self.setGeometry(100, 100, 300, 300)

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
                        font-weight: bold;
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
                        font-weight: bold;          /* Font weight (bold, normal, etc.) */
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
                        font-weight: bold;          /* Font weight (bold, normal, etc.) */
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
                        font-weight: bold;
                    }
                """)

        # print("Current working directory:", os.getcwd())

        # Get the directory of the current script and its parent
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.script_dir)
        self.config_file = os.path.join(self.parent_dir, "config.json")

        # Load configuration
        self.config = self.load_config(self.config_file)

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
        self.field_size = 1000
        self.use_file_upload = False
        self.num_iterations = 5000

        # Initialize the UI
        self.init_ui()


    def load_config(self, file_path):
        """Load configuration from a JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file '{file_path}' not found. Using default values.")
            return {}
        except json.JSONDecodeError:
            print(f"Config file '{file_path}' is not a valid JSON. Using default values.")
            return {}

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
        layout.addWidget(QLabel("Number of Lineups:", self), 0, 0)  # Row 0, Column 0
        self.num_lineups_input = QLineEdit(str(self.num_lineups), self)
        layout.addWidget(self.num_lineups_input, 0, 1)  # Row 0, Column 1

        layout.addWidget(QLabel("Field Size:", self), 1, 0)  # Row 0, Column 2
        self.field_size_input = QLineEdit(str(self.field_size), self)
        layout.addWidget(self.field_size_input, 1, 1)  # Row 0, Column 3

        # Row 2
        layout.addWidget(QLabel("Use Contest Data:", self), 0, 2)
        self.use_contest_data_checkbox = QCheckBox(self)
        self.use_contest_data_checkbox.setChecked(self.use_contest_data)
        layout.addWidget(self.use_contest_data_checkbox, 0, 3)

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

        central_widget.setLayout(layout)

    def update_parameters(self):
        # Retrieve the current values from the input fields
        try:
            self.site = "dk"
            self.num_lineups = int(self.num_lineups_input.text())
            self.num_uniques = int(self.num_uniques_input.text())
            self.use_contest_data = self.use_contest_data_checkbox.isChecked()
            self.field_size = int(self.field_size_input.text())
            #self.use_file_upload = self.use_file_upload_checkbox.isChecked()
            self.num_iterations = int(self.num_iterations_input.text())
            self.randomness = self.randomness_amount_input.value()
            self.min_salary = self.min_salary_input.value()
            self.projection_minimum = self.projection_minimum_input.value()
            self.global_team_limit = self.global_team_limit_input.value()

            QMessageBox.information(self, "Success", "Parameters updated successfully!")
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
            sim = nba_gpp_simulator.NBA_GPP_Simulator(self.site, self.field_size, self.num_iterations,
                                                      self.use_contest_data, self.use_file_upload, self.min_salary,
                                                      self.projection_minimum)
            sim.generate_field_lineups()
            sim.run_tournament_simulation()
            sim.output()
            QMessageBox.information(self, "Success", "Tournament simulation completed!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def run_swap_sim(self):
        self.update_parameters()

        # Get the directory of the current script
        if getattr(sys, 'frozen', False):  # Running in PyInstaller bundle
            script_dir = sys._MEIPASS
        else:  # Running in development
            script_dir = os.path.dirname(os.path.abspath(__file__))

        run_swap_sim_path = os.path.join(script_dir, 'run_swap_sim.py')

        # Debugging: Print the script paths
        print(f"Script Directory: {script_dir}")
        print(f"Run Swap Sim Path: {run_swap_sim_path}")

        # Ensure the script exists
        if not os.path.exists(run_swap_sim_path):
            print(f"Error: Script not found at {run_swap_sim_path}")
            return

        # Set up QProcess
        working_dir = os.path.dirname(run_swap_sim_path)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(working_dir)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.handle_process_error)

        # Use the system Python interpreter
        python_executable = sys.executable  # System Python in PATH

        # Build the command
        command = (
            python_executable,  # Python interpreter
            run_swap_sim_path,  # Script path
            str(self.num_iterations),  # num_iterations argument
            self.site,  # site argument
            str(self.num_uniques),  # num_uniques argument
            str(self.num_lineups),
            str(self.min_salary),
            str(self.projection_minimum)
        )

        # Debugging: Print the command
        print("Command to execute:", command)

        # Start the process with all arguments
        self.process.start(command[0], command[1:])


    def handle_stdout(self):
        output = self.process.readAllStandardOutput().data().decode().strip()
        print(f"STDOUT: {output}")

    def handle_stderr(self):
        error = self.process.readAllStandardError().data().decode().strip()
        print(f"STDERR: {error}")

    def on_process_finished(self, exit_code, exit_status):
        print(f"Process finished with exit code {exit_code} and status {exit_status}")

    def handle_process_error(self, error):
        print(f"Process error occurred: {error}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Necessary for Windows
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec())

