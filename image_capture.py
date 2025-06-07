import tkinter as tk
import customtkinter
import cv2
from PIL import Image, ImageTk
import os
import time

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

class CameraCapture:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Capture Image")
        self.window.configure(bg='gray20')  # Dark background color

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(self.window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT), bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.btn_snapshot = customtkinter.CTkButton(self.window, text="CAPTURE", 
                                                    command=self.snapshot)
        self.btn_snapshot.grid(row=1, column=0, padx=10, pady=10)

        self.btn_close = customtkinter.CTkButton(self.window, text="Close", fg_color="red", 
                                                 command=self.close_window)
        self.btn_close.grid(row=1, column=1, padx=10, pady=10)

        self.status_label = tk.Label(self.window, text="", fg='white', bg='gray20')
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.update()

        self.save_folder = "image_capture"
        os.makedirs(self.save_folder, exist_ok=True)

    def close_window(self):
        self.window.destroy()

    def snapshot(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                filename = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
                filepath = os.path.join(self.save_folder, filename)
                cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.status_label.config(text=f"Image captured and saved as {filename}")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(50, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x400")
    app = CameraCapture(root)
    root.mainloop()
