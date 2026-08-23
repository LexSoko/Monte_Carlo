import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
class PlotUpdater:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Plotting")

        self.fig, self.axs = plt.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initial plots
        self.line1, = self.axs[0, 0].plot([], [], ".-")
        self.line2, = self.axs[0, 1].plot([])
        self.line3, = self.axs[1, 0].plot([])
        self.line4, = self.axs[1, 1].plot([])

        self.text1 = self.axs[0, 0].text(0.9, 0.9, "")
        self.text2 = self.axs[0, 0].text(0.1, 0.9, "")
        self.text3 = self.axs[0, 0].text(0.1, 0.7, "")

        self.simulation_running = False

    def start_simulation(self):
        self.simulation_running = True
        threading.Thread(target=self.run_simulation).start()

    def stop_simulation(self):
        self.simulation_running = False

    def run_simulation(self):
        current_array = np.random.rand(2, 100)
        all_energies_k = np.random.rand(100)
        all_energies_avg = np.random.rand(100)
        all_energies_var = np.random.rand(100)

        while self.simulation_running:
            # Update data here
            current_array = np.random.rand(2, 100)
            all_energies_k = np.random.rand(100)
            all_energies_avg = np.random.rand(100)
            all_energies_var = np.random.rand(100)
            current_T = np.random.rand()
            n = np.random.randint(0, 100)
            same = np.random.randint(0, 100)

            self.update_plot(current_array, all_energies_k, all_energies_avg, all_energies_var, current_T, n, same)
            time.sleep(0.1)  # Adjust the sleep time as needed

    def update_plot(self, current_array, all_energies_k, all_energies_avg, all_energies_var, current_T, n, same):
        self.line1.set_data(current_array.T[0], current_array.T[1])
        self.line2.set_ydata(all_energies_k)
        self.line3.set_ydata(all_energies_avg)
        self.line4.set_ydata(all_energies_var)
        self.text1.set_text(f"{current_T:.3f}")
        self.text2.set_text(f"{n:.0f}")
        self.text3.set_text(f"{same:.0f}")

        # Rescale the axis limits
        self.axs[0, 0].relim()
        self.axs[0, 0].autoscale_view()
        self.axs[0, 1].relim()
        self.axs[0, 1].autoscale_view()
        self.axs[1, 0].relim()
        self.axs[1, 0].autoscale_view()
        self.axs[1, 1].relim()
        self.axs[1, 1].autoscale_view()

        self.canvas.draw()
if __name__ == "__main__":
    root = tk.Tk()
    app = PlotUpdater(root)

    control_frame = ttk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    start_button = ttk.Button(control_frame, text="Start Simulation", command=app.start_simulation)
    start_button.pack(side=tk.LEFT)

    stop_button = ttk.Button(control_frame, text="Stop Simulation", command=app.stop_simulation)
    stop_button.pack(side=tk.LEFT)

    root.mainloop()
