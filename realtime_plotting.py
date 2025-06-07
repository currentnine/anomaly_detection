import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import re
import  constants as const

# Suppress the tight_layout warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class RealTimePlotter(tk.Frame):
    def __init__(self, parent,file_path,update_interval=1000):
        super().__init__(parent)
        self.running = False
        self.file_path = file_path
        self.update_interval = update_interval
        self.plot_initialized = False  # Indicator for the initial plot

        self.fig = Figure(figsize=(6, 4))  # Adjust figsize to match your aspect ratio
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        print('--------------[RealTimePlotter]--------------')
        self.update_plot()
    def start_plot(self):
        print("Start updating the plot.")
        self.running = True
        self.update_plot()

    def stop_plot(self):
        print("Stop updating the plot.")
        self.running = False
        
    def update_plot(self):
        if self.running:
            data = self.read_data_from_file()
            self.ax.clear()
            if not data.empty:
                self.ax.plot(data['epoch'], data['loss'], marker='o', linestyle='-')
                self.ax.set_xlabel('Epoch')
                self.ax.set_ylabel('Loss')
                self.ax.set_title('Loss over Epochs')
                
                # Apply tight_layout only once after first plot initialization
                if not self.plot_initialized:
                    self.fig.tight_layout()
                    self.plot_initialized = True

            self.canvas.draw()
            self.after(self.update_interval, self.update_plot)
        else:
            print("Plotting stopped.")

    def read_data_from_file(self):
        data = {'epoch': [], 'loss': []}
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    parsed_data = self.parse_line(line)
                    if parsed_data:
                        data['epoch'].append(parsed_data[0])
                        data['loss'].append(parsed_data[1])
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        return pd.DataFrame(data)


    @staticmethod
    def parse_line(line):
        match = re.search(r'Epoch (\d+) - Step \d+: loss = ([\-\d\.]+)', line)
        if match:
            return int(match.group(1)), float(match.group(2))
        return None
