import tkinter as tk
from tkinter import ttk

class NBA_DFS_UI:
    def __init__(self, master):
        self.master = master
        self.master.title("NBA DFS Simulator")

        # Input fields
        self.iterations_label = ttk.Label(master, text="Number of Iterations:")
        self.iterations_label.grid(row=0, column=0, padx=10, pady=10)
        self.iterations_entry = ttk.Entry(master)
        self.iterations_entry.grid(row=0, column=1, padx=10, pady=10)

        self.simulate_button = ttk.Button(master, text="Run Simulations", command=self.run_simulation)
        self.simulate_button.grid(row=1, column=0, columnspan=2, pady=20)

        # Output
        self.output_text = tk.Text(master, height=20, width=50)
        self.output_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def run_simulation(self):
        num_iterations = int(self.iterations_entry.get())
        self.output_text.insert(tk.END, f"Running {num_iterations} simulations...\n")
        # Replace this with your simulation logic
        self.output_text.insert(tk.END, "Simulation completed!\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = NBA_DFS_UI(root)
    root.mainloop()
