# def decimal_to_percentage(decimal_number):
#     decimal_number= int(decimal_number)
#     percentage = decimal_number * 100
#     return f'{percentage:.0f}%'

# # Example usage
# if __name__ == "__main__":
#     decimal_value = '0.8206559665038381'
#     percentage_value = decimal_to_percentage(decimal_value)
#     print(f"Decimal: {decimal_value}, Percentage: {percentage_value}")

def increase_threshold_by_percentage(percentage):
    try:
        threshold = float(consts.THRESHOLD)
        if percentage < 0:
            return "Percentage should be a positive number."
        new_threshold = threshold * (1 + percentage / 100)
        return new_threshold
    except ValueError:
        return "Invalid input. Please provide valid numeric values."

# Example usage
if __name__ == "__main__":
    import constants as consts  # Make sure to import the module containing your consts.THRESHOLD value
    
    percentage = 50
    new_threshold = increase_threshold_by_percentage(percentage)
    print(f"Original Threshold: {consts.THRESHOLD}")
    print(f"New Threshold after {percentage}% increase: {new_threshold}")






import tkinter as tk
from tkinter import ttk
import customtkinter

# Assuming consts.THRESHOLD is already defined somewhere in your code

class YourApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        # ... Other initialization code ...

        config_frame5 = ttk.Frame(self)
        config_frame5.grid(row=0, column=0, padx=5, pady=5)

        custom_font = ('Arial', 12)  # Example font

        # Threshold Label
        thrushold_label = ttk.Label(config_frame5, style="Custom.TLabel", text="THRESHOLD")
        thrushold_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")

        # Threshold Entry
        self.TH_entry = ttk.Entry(config_frame5, width=20)
        self.TH_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        self.update_entry_with_threshold()

        # Threshold Confirm Button
        self.thrushold_btn = customtkinter.CTkButton(config_frame5, font=custom_font, text="CONFIRM", command=self.change_Trushold)
        self.thrushold_btn.grid(row=8, column=3, padx=(10, 10), pady=(10, 10), sticky="w")

        # Threshold Slider
        self.slider_1 = customtkinter.CTkSlider(config_frame5, from_=0, to=100, number_of_steps=100, command=self.slider_changed)
        self.slider_1.grid(row=9, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.slider_1.set(consts.THRESHOLD * 100)

    def update_entry_with_threshold(self):
        # Update entry with threshold value as percentage
        self.TH_entry.delete(0, tk.END)
        self.TH_entry.insert(0, str(round(consts.THRESHOLD * 100)) + '%')

    def slider_changed(self, event=None):
        # Update consts.THRESHOLD and entry when slider changes
        print(self.slider_1.get() / 100)
        consts.THRESHOLD = self.slider_1.get() / 100
        self.update_entry_with_threshold()

    def change_Trushold(self):
        # Update consts.THRESHOLD and slider when entry changes
        try:
            threshold_percentage = float(self.TH_entry.get().strip('%'))
            print(threshold_percentage)
            consts.THRESHOLD = threshold_percentage / 100
            self.slider_1.set(threshold_percentage)
        except ValueError:
            # Handle invalid input
            self.update_entry_with_threshold()

# Create and run application
app = YourApplication()
app.mainloop()
