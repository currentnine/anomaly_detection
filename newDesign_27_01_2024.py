# from main_test_all_v2 import hi
import tkinter as tk
import customtkinter
from realtime_plotting1 import RealTimePlotter
from tkinter import ttk, filedialog, messagebox, font
from PIL import Image, ImageTk
import subprocess
import os
from tkinter import filedialog
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")
import constants as consts
import importlib
import imghdr
import threading
import shutil
from image_capture import CameraCapture


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        self.title("Education Software")
        self.attributes('-fullscreen', True)
        self.inspection_processes = [] 
        self.constantPath =consts.__file__
        self.current_dir = os.path.dirname(os.path.abspath(__file__))  
        
        self.checkpoint_path=consts.CHECKPOINT_PATH
        weight_path='_'
        self.train=consts.TRAIN_STATUS #bool variable for training status 
        # Initialize variables to track if Good Data and Bad Data are selected
        self.data_selected = False
        custom_font = customtkinter.CTkFont( size=16)
        # self.bad_data_selected = False
        # Sidebar
        self.initialisation_folder()
        # output_path=consts.OUTPUT_FILE_PATH
        output_path = self.initialisation_folder()
        
        sidebar = customtkinter.CTkFrame(self, width=200)
        sidebar.pack(side="left", fill="y", padx=5, pady=5)

        # Train button
        train_button = customtkinter.CTkButton(sidebar, font=custom_font,text="Train", command=lambda: self.on_tab_selected("train") )
        train_button.pack(pady=10, fill="x")

        # Test button
        test_button = customtkinter.CTkButton(sidebar, font=custom_font,text="Test", command=lambda: self.on_tab_selected("test") )
        test_button.pack(pady=10, fill="x")

        # Bottom frame for exit button
        bottom_frame = customtkinter.CTkFrame(sidebar)
        bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Exit button
        exit_button = customtkinter.CTkButton(bottom_frame, fg_color="red", font=custom_font,text="Exit", command=self.quit_application  )
        exit_button.pack(pady=10)

        # Content area
        self.content_frame = customtkinter.CTkFrame(self)
        self.content_frame.pack(side="right", fill="both", expand=True)

        # Initialize tab frames (hidden by default)
        self.train_frame = customtkinter.CTkFrame(self.content_frame)
        self.test_frame = customtkinter.CTkFrame(self.content_frame)
        tab1 = self.train_frame 
        
        tab2 = self.test_frame

        # Initially select the train tab
        self.on_tab_selected("test")

        # Train Frame 
        # Place buttons in row 0
        self.button_start = customtkinter.CTkButton(tab1, font=custom_font ,text="Start",command=self.start_training,  state=tk.DISABLED)
        self.button_start.grid(row=0, column=0, padx=10, pady=10 ,sticky="nw")

        self.button_stop = customtkinter.CTkButton(tab1, font=custom_font,text="Stop",command=self.stop_training, state=tk.DISABLED)
        self.button_stop.grid(row=0, column=1, padx=10, pady=10 ,sticky="nw")

        self.button_good_dataset = customtkinter.CTkButton(tab1, font=custom_font,text="Select Dataset file", command=self.select_dataset_path)
        self.button_good_dataset.grid(row=0, column=2, padx=10, pady=10 ,sticky="nw")

        self.button_restart = customtkinter.CTkButton(tab1, font=custom_font,text="Restart",command=self.Restart_train,  state=tk.DISABLED)
        self.button_restart.grid(row=0, column=3, padx=10, pady=10 ,sticky="nw")

        self.button_clear = customtkinter.CTkButton(tab1, font=custom_font,text="Clear & Reinitialize",command=self.Clear_train)
        self.button_clear.grid(row=0, column=4, padx=10, pady=10 ,sticky="nw")
        # tab1.grid_rowconfigure(1, weight=1)
        # tab1.grid_rowconfigure(2, weight=1)
        weights = [0, 0, 0,1]  # For example, three rows with different weights
        for row, weight in enumerate(weights):
            
            tab1.grid_rowconfigure(row, weight=weight)

        
        self.update_start_button_state()

        # Assigning weights to columns
        weights = [0, 0, 0,0,1]  # Total weight = 6
        for col, weight in enumerate(weights):
                tab1.grid_columnconfigure(col, weight=weight)

        tab1
        
        # Place CTkTextbox in the first column and full vertical space from the second row
        matrix_green = "#F8F8F8"
        self.training_output = customtkinter.CTkTextbox(tab1,width=500,font=custom_font ,text_color=matrix_green)
        self.training_output.grid(row=1, column=0, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.update_gui_with_file_content()
        style = ttk.Style()
        style.configure("Custom.TLabel", font=('Helvetica', 16)) 
        style.configure("Custom.TLabelframe.Label", font=('Helvetica', 16,'bold'))
        #-------------"Grid 1" - Configuration Frame----------
        config_frame = ttk.LabelFrame(tab1,style="Custom.TLabelframe",text="Configuration")
        # Label and Entry for BATCH_SIZE
        batch_size_label = ttk.Label(config_frame, style="Custom.TLabel",text="BATCH_SIZE")
        batch_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry1 = ttk.Entry(config_frame, width=20)
        self.entry1.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.entry1.insert(0, consts.BATCH_SIZE)  # Set BATCH_SIZE value

        # Label and Entry for NUM_EPOCHS
        num_epochs_label = ttk.Label(config_frame, style="Custom.TLabel",text="NUM_EPOCHS")
        num_epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry2 = ttk.Entry(config_frame, width=20)
        self.entry2.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry2.insert(0, consts.NUM_EPOCHS)  # Set NUM_EPOCHS value
        # Label and Entry for NUM_EPOCHS
        input_size_label = ttk.Label(config_frame, style="Custom.TLabel",text="TRAINING_INPUT_SIZE")
        input_size_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry3 = ttk.Entry(config_frame, width=20)
        self.entry3.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.entry3.insert(0, consts.INPUT_SIZE)  # Set NUM_EPOCHS value
        # Add a check button at the bottom
        #check_button = ttk.Button(config_frame, font=custom_font,text="Confirm", command=self.check_action)
        check_button = customtkinter.CTkButton(config_frame,text="CONFIRM", command=self.check_action)
        check_button.grid(row=3, columnspan=2 ,padx=5, pady=10)

        # Place the LabelFrame within the grid
        # config_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nw")
        config_frame.grid(row=1, column=4,sticky="nw" ,padx=10, pady=10)

        #-------------"Grid 2"----------
        config_frameG2 = ttk.LabelFrame(tab1, style="Custom.TLabelframe",text="DISPLAY TRAINING PARAMETERS")
        #BATCH_SIZE
        
        self.batch_size_label = ttk.Label(config_frameG2, style="Custom.TLabel",text="BATCH_SIZE : "+str(consts.BATCH_SIZE))
        self.batch_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # self.batch_size_labelValue = ttk.Label(config_frameG2, style="Custom.TLabel",text=consts.BATCH_SIZE)
        # self.batch_size_labelValue.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        #NUM_EPOCHS
        self.num_epochs_label_D = ttk.Label(config_frameG2, style="Custom.TLabel",text="NUM_EPOCHS : "+str(consts.NUM_EPOCHS))
        self.num_epochs_label_D.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        # self.num_epochs_label_Value = ttk.Label(config_frameG2, style="Custom.TLabel",text=consts.NUM_EPOCHS)
        # self.num_epochs_label_Value.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        # LEARNINGN RATE
        LR_Label_D = ttk.Label(config_frameG2, style="Custom.TLabel",text="LEARNINGN RATE : "+str(consts.LR))
        LR_Label_D.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        # self.LR_Label_Value = ttk.Label(config_frameG2, style="Custom.TLabel",text=consts.LR)
        # self.LR_Label_Value.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        # IMAGES SIZE
        sizeInput=str(consts.INPUT_SIZE)
        self.IMAGES_SIZE_D = ttk.Label(config_frameG2, style="Custom.TLabel",text="TRAINIG_INPUT_SIZE : "+sizeInput+' x '+sizeInput)
        self.IMAGES_SIZE_D.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        # self.IMAGES_SIZE_D_Value = ttk.Label(config_frameG2, style="Custom.TLabel",text=sizeInput+' x '+sizeInput)
        # self.IMAGES_SIZE_D_Value.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        # DATASET GOOD
        self.DATASET_GOOD = ttk.Label(config_frameG2, style="Custom.TLabel",text=" DATASET PATH  : _")
        self.DATASET_GOOD.grid(row=4, column=0, columnspan=3,padx=5, pady=5, sticky="w")
        # self.DATASET_GOOD_path = ttk.Label(config_frameG2,text='_')
        # self.DATASET_GOOD_path.grid(row=4, column=1, padx=5, pady=5, sticky="w")
 
        # fastflow_experiment_checkpointsstyle="Custom.TLabel"
        self.checkpoints_D = ttk.Label(config_frameG2, style="Custom.TLabel",text="CHECKPOINTS: ")
        self.checkpoints_D.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        # self.checkpoints_path = ttk.Label(config_frameG2, text="_")
        # self.checkpoints_path.grid(row=5, column=1, padx=5,columnspan=3, pady=5, sticky="w")

        #experiment 
        
        experimentInedx=len(os.listdir(consts.CHECKPOINT_DIR))
        self.exp_D = ttk.Label(config_frameG2, style="Custom.TLabel",text="ACTUAL EXPERIMENT NUMBER  : "+str(experimentInedx))
        self.exp_D.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        # self.exp_V = ttk.Label(config_frameG2, style="Custom.TLabel",text=experimentInedx)
        # self.exp_V.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        
        # self.get_last_experiment(checkpoint_path)
        #----
        # config_frameG2.grid(row=1, column=1, padx=10, pady=10, sticky="nw")
        config_frameG2.grid(row=1, column=1 ,columnspan=3,sticky="nsew", padx=10, pady=10)

        #-------------"Grid 3"----------
        self.reload_constants
        self.real_time_plotter = None
        self.setup_progress_window(consts.NUM_EPOCHS)
        self.initialize_real_time_plotter(output_path)
        
        

         #configuration tab2
        self.CAPTURE_BTN = customtkinter.CTkButton(tab2, font=custom_font,text="CAPTURE",command=self.caputre_image)
        self.CAPTURE_BTN.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), sticky="w")
        self.LOAD_W_1TEST_BTN = customtkinter.CTkButton(tab2, font=custom_font,text="LOAD WIEGHT",command=self.laod_wieght)
        self.LOAD_W_1TEST_BTN.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="w")
        self.LOAD_IMAGE_BTN = customtkinter.CTkButton(tab2, font=custom_font,text="LOAD IMAGE",command=self.load_image)
        self.LOAD_IMAGE_BTN.grid(row=2, column=2, padx=(10, 10), pady=(10, 10), sticky="w")

        self.START_SINGLE_TEST_BTN = customtkinter.CTkButton(tab2, font=custom_font,text="START SINGLE TEST ",command=self.start_test)
        self.START_SINGLE_TEST_BTN.grid(row=2, column=3, padx=(10, 10), pady=(10, 10), sticky="w")
        self.START_SINGLE_TEST_BTN = customtkinter.CTkButton(tab2, font=custom_font,text="START MULTIPLE TEST ",command=self.start_test_All)
        self.START_SINGLE_TEST_BTN.grid(row=2, column=4, padx=(10, 10), pady=(10, 10), sticky="w")

        # config_frame4 = ttk.LabelFrame(tab2,style="Custom.TLabelframe",text="START ALL TEST & CALCULATE THRUSHOLD")
        config_frame4 = ttk.LabelFrame(tab2,style="Custom.TLabelframe",text="CALCULATE THRESHOLD")
        # Label and Entry for BATCH_SIZE
        # check_button = customtkinter.CTkButton(config_frame4,font=custom_font ,text="Select weight ", command=self.laod_wieght)
        # check_button.grid(row=0, column=1, padx=5, pady=10)
        check_button = customtkinter.CTkButton(config_frame4, font=custom_font,text="CALCULATE THRESHOLD", command=self.start_th)
        check_button.grid(row=0, column=1, padx=5, pady=10)
        # acc = ttk.Label(config_frame4, style="Custom.TLabel",text="accuracy_score : ")
        # acc.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        # self.acc_v = ttk.Label(config_frame4, style="Custom.TLabel",text='_')
        # self.acc_v.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        # self.check_for_update()

        #Weight selected
        weight_L=ttk.Label(config_frame4, style="Custom.TLabel",text="Weight_Selected : ")
        weight_L.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.weight_v = ttk.Label(config_frame4, font=custom_font,text=weight_path)
        self.weight_v.grid(row=6, column=2, padx=5, pady=5, sticky="w")

        
        
        # Place the LabelFrame within the grid
        config_frame4.grid(row=3, column=0, columnspan=3, sticky="nw", padx=10, pady=10)

        config_frame5 = ttk.LabelFrame(tab2,style="Custom.TLabelframe",text="")
        
        thrushold_label = ttk.Label(config_frame5, style="Custom.TLabel",text="THRUSHOLD")
        thrushold_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")

        self.threshold_percentage_label = ttk.Label(config_frame5, text="", style="Custom.TLabel")
        self.threshold_percentage_label.grid(row=8, column=1, padx=5, pady=5, sticky="w")

        self.TH_entry = ttk.Entry(config_frame5, width=20)
        self.TH_entry.grid(row=8, column=2, padx=5, pady=5, sticky="w")

        self.TH_entry.insert(0, consts.THRESHOLD)
        self.thrushold_btn = customtkinter.CTkButton(config_frame5, font=custom_font,text="CONFIRM",command=self.change_Trushold)
        self.thrushold_btn.grid(row=8, column=3, padx=(10, 10), pady=(10, 10), sticky="w")
        # -----------
        self.slider_1 = customtkinter.CTkSlider(config_frame5, from_=0, to=100, number_of_steps=100)
        self.slider_1.set(100)
        self.slider_1.grid(row=9, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        
        # ---------------
        config_frame5.grid(row=4, column=0, columnspan=3, sticky="nw", padx=10, pady=10)

        #
        # 
        # 
        # 
        #  Image and predection----------------,font=("Arial", 12, "bold")
        fixed_size = (512, 512)  
        background_color = self.cget("background")
        self.input_test_label = ttk.Label(tab2, font=custom_font,text="Input Image", compound="bottom",foreground="white", background=background_color)
        inImageTest1=consts.IMAGE_TEST_PATH
        image = Image.open(inImageTest1)  # Replace with your image path
        image = image.resize(fixed_size,  Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.input_test_label.config(image=photo)
        self.input_test_label.image = photo
        self.input_test_label.grid(row=5, column=0, padx=0, pady=10, sticky="nsew")
        self.input_test_label.bind("<Button-1>", lambda event: self.load_image())
        self.input_test_label.bind("<Enter>", lambda event: self.input_test_label.config(cursor="hand2"))
        self.input_test_label.bind("<Leave>", lambda event: self.input_test_label.config(cursor=""))

        

        # self.input_test_label.config(text="Prediction Score: " + str(93))
        inImageTest2=consts.IMAGE_PREDECTION_PATH#,font=("Arial", 12, "bold")
        self.predection_label = ttk.Label(tab2, font=custom_font,text="Predection", compound="bottom",foreground="white", background=background_color)
        self.predection_label.grid(row=5, column=4, padx=0, pady=10, sticky="nsew")
        image = Image.open(inImageTest2)  # Replace with your image path
        image = image.resize(fixed_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.predection_label.config(image=photo)
        self.predection_label.image = photo

    def check_and_display_score(self):
        # Check if SCORE_TEST exists and is not None
        print("im hereeeeeeeeeeeeee")
        if hasattr(consts, 'SCORE_TEST') and consts.SCORE_TEST is not None:
            # Update the label with the score
            self.input_test_label.config(text=f"Score: {consts.SCORE_TEST}")    

    def change_Trushold(self):
        threshold_value = self.TH_entry.get()
        self.update_value(self.constantPath,'THRESHOLD',threshold_value)
        self.reload_constants()
        if str(consts.THRESHOLD) == str(threshold_value):
            messagebox.showinfo("Update Successful", f"The threshold value has been successfully updated to {threshold_value}.")
        else:
            messagebox.showerror("Update Failed", "Failed to update the threshold value.")

    def reload_constants(self):
        importlib.reload(consts)

    # def check_for_update(self):
    #     self.reload_constants()
    #     current_value = consts.SCORE_ALL

    #     # Check if acc_v is initialized
    #     if hasattr(self, 'acc_v'):
    #         self.acc_v.config(text=str(current_value))
            
    #     else:
    #         print("acc_v is not initialized")
    #     self.after(1000, self.check_for_update)
           
    def laod_wieght(self):
        global weight_path
        weight_path = filedialog.askopenfilename()

        # Check if the selected file is a .pt file
        if weight_path.endswith('.pt'):
            self.update_constant_file('WEIGHT_PATH', weight_path)
            self.weight_v.configure(text=weight_path)
            print(weight_path)
            messagebox.showinfo("Success", "Weight file loaded successfully.")
        else:
         messagebox.showerror("Invalid File", "The selected file is not a valid .pt weight file.")        
    
    def setup_progress_window(self,total_epochs):
        
        self.progress = ttk.Progressbar(self.train_frame , orient="horizontal", mode='determinate', maximum=total_epochs)
        self.progress.grid(row=2,  column=1,columnspan=4, sticky="nsew", padx=10, pady=10)

        self.update_progress_bar()

    def update_progress_bar(self):
        global output_path

        with open(output_path, "r") as file:
            lines = file.readlines()

        current_epoch = 0
        for line in lines:
            if "Epoch" in line:
                parts = line.split()
                current_epoch = int(parts[1])
        # print('current_epoch : ',current_epoch)
        self.progress['value'] = current_epoch
        self.after(1000, self.update_progress_bar)
    def initialize_real_time_plotter(self,file_path):
        print('initialize_real_time_plotter output ==>',file_path)
        # Remove the existing RealTimePlotter instance
        if self.real_time_plotter is not None:
            self.real_time_plotter.grid_forget()  # This removes the widget from the layout
        # Create a new RealTimePlotter instance
        self.real_time_plotter = RealTimePlotter(self.train_frame ,file_path)  # Replace with appropriate arguments

        # Place the new instance in the grid
        self.real_time_plotter.grid(row=3, column=1, columnspan=4, sticky="nsew")

    def select_dataset_path(self):
        # Open a dialog to select a folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            # Check if 'test' and 'train' folders exist in the selected directory
            required_folders = ['test', 'train']
            if all(os.path.isdir(os.path.join(folder_selected, folder)) for folder in required_folders):
                # Update the constant in constant.py
                if self.update_constant_file('DATASET_PATH', folder_selected):
                    self.DATASET_GOOD.configure(text=" DATASET PATH  : "+folder_selected)
                    self.data_selected = True
                    # Notify the user of successful update
                    messagebox.showinfo("Update Successful", " Dataset path Selected successfully.")
                    self.update_start_button_state()
                    return self.data_selected
                else:
                    # Notify the user of an error
                    self.content_frame.configure(cursor="")
                    self.data_selected = False
                    messagebox.showerror("Update Error", "Failed to Select  Dataset path.")
                    return self.data_selected
            else:
                # Notify the user that the selected folder does not contain the required subfolders
                self.content_frame.configure(cursor="")
                messagebox.showerror("Invalid Folder", "The selected folder does not contain 'test' and 'train' subfolders.")
                return self.data_selected
    
    def select_test_all_path(self):
        # Open a dialog to select a folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            # Check if 'test' and 'train' folders exist in the selected directory
            required_folders = ['bad', 'good']
            if all(os.path.isdir(os.path.join(folder_selected, folder)) for folder in required_folders):
                # Update the constant in constant.py
                if self.update_constant_file('MULTIPLE_TEST_PATH', folder_selected):
                    # Notify the user of successful update
                    messagebox.showinfo("Update Successful", " TEST path Selected successfully.")
                    self.update_start_button_state()
                    return True
                else:
                    # Notify the user of an error
                    self.content_frame.configure(cursor="")
                    self.data_selected = False
                    messagebox.showerror("Update Error", "Failed to Select  TEST path.")
                    return False
            else:
                # Notify the user that the selected folder does not contain the required subfolders
                self.content_frame.configure(cursor="")
                messagebox.showerror("Invalid Folder", "The selected folder does not contain 'bad' and 'good' subfolders.")
                return False
            
    def select_dataset_path_good(self):
        # Open a dialog to select a folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            # Check if 'train' folder exists in the selected directory
            train_folder = os.path.join(folder_selected, 'train')
            if os.path.isdir(train_folder):
                # Update the constant in constant.py and the GUI
                if self.update_constant_file('DATASET_PATH', folder_selected):
                    self.DATASET_GOOD.configure(text=" DATASET PATH  : "+folder_selected)
                    self.data_selected = True
                    # Notify the user of successful update
                    messagebox.showinfo("Update Successful", "Dataset path selected successfully.")
                    self.update_start_button_state()
                    return self.data_selected
                else:
                    # Notify the user of an error
                    self.content_frame.configure(cursor="")
                    self.data_selected = False
                    messagebox.showerror("Update Error", "Failed to select dataset path.")
                    return self.data_selected
            else:
                # Notify the user that the selected folder does not contain the 'train' subfolder
                self.content_frame.configure(cursor="")
                messagebox.showerror("Invalid Folder", "The selected folder does not contain a 'train' subfolder.")
                return self.data_selected         
    def update_constant_file(self, constant_name, new_value, add_if_not_exist=False):
        print('Start updating constant file')
        new_value = new_value.replace("\\", "/")

        if not os.path.exists(self.constantPath):
            print(f"Error: '{self.constantPath}' file not found.")
            return False

        try:
            # Read the current contents of the file and log the old value
            with open(self.constantPath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            old_value = None
            for line in lines:
                if line.strip().startswith(constant_name):
                    old_value = line.strip()
                    break

            # print(f"Old value for '{constant_name}': {old_value}")

            # Update the specific constant line
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(constant_name):
                    lines[i] = f'{constant_name} = "{new_value}"\n'
                    updated = True
                    break

            if not updated and add_if_not_exist:
                lines.append(f'{constant_name} = "{new_value}"\n')
                updated = True

            if not updated:
                print(f"Constant '{constant_name}' not found in the file.")
                return False

            # Write the updated lines back to the file
            with open(self.constantPath, 'w', encoding='utf-8') as file:
                file.writelines(lines)

            # Verify the change by reading the file again
            with open(self.constantPath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            new_value_in_file = None
            for line in lines:
                if line.strip().startswith(constant_name):
                    new_value_in_file = line.strip()
                    break

            # print(f"New value for '{constant_name}' in file: {new_value_in_file}")

            return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False


    def update_start_button_state(self):
        # if self.good_data_selected and self.bad_data_selected:
        if self.data_selected :
            self.button_start.configure(state=tk.NORMAL)
            self.button_stop.configure(state=tk.NORMAL)
            self.button_restart.configure(state=tk.NORMAL)
        else:
            self.button_start.configure(state=tk.DISABLED)
            self.button_stop.configure(state=tk.DISABLED)
            self.button_restart.configure(state=tk.DISABLED)

    def quit_application(self):
        self.stop_training()
        self.quit()
        print("EXIT and Stop Training")

    def on_tab_selected(self, tab_name):
        if tab_name == "train":
            self.test_frame.pack_forget()
            self.train_frame.pack(fill="both", expand=True)
            # Update train frame content
        elif tab_name == "test":
            self.train_frame.pack_forget()
            self.test_frame.pack(fill="both", expand=True)
    
    def initialisation_folder(self):
        global experimentInedx
        global output_path
        global weight_path
        print('---------------start initialisation-------------')
        #creat new file for eash training
        os.makedirs(consts.CHECKPOINT_DIR, exist_ok=True)
        self.experimentInedx=(len(os.listdir(consts.CHECKPOINT_DIR))+1)
        id=self.experimentInedx
        experimentInedx=id
        #creat the checkpoint_dir path and experiment folder
        checkpoint_path = os.path.join(
            consts.CHECKPOINT_DIR, "exp%d" % id  )
        checkpoint_path= os.path.abspath(checkpoint_path)
        # update the CHECKPOINT_PATH
        self.update_constant_file('CHECKPOINT_PATH',checkpoint_path)
        print('update checkpoitn path in constants : ',checkpoint_path)
        #creat  experiment folder
        os.makedirs(checkpoint_path, exist_ok=True)
        print('creat file : ',checkpoint_path)
        #creat the path of output.txt
        file_path= os.path.join(checkpoint_path, 'output.txt')
        
        #create output.txt file in experiment folder
        if not os.path.exists(file_path):
        # Create the file output.txt
            with open(file_path, 'w') as file:
                file.write("")
                self.update_constant_file('OUTPUT_FILE_PATH',file_path)  
                self.reload_constants()
                output_path=file_path
        self.update_value(self.constantPath,'SCORE_ALL','_')
        self.update_constant_file('IMAGE_TEST_PATH',os.path.join(self.current_dir, "inputE.png"))
        self.update_constant_file('IMAGE_PREDECTION_PATH',os.path.join(self.current_dir, "waiting.png"))
        self.update_value(self.constantPath,'PREDECTION_RESUILT',"No result...")
        weight_path='_'
        self.update_constant_file('WEIGHT_PATH', weight_path)
        self.update_constant_file('DATASET_PATH', '_')   
        self.update_constant_file('SCORE_TEST', '_')   
        self.update_constant_file('MULTIPLE_TEST_PATH', '_')   

        print('output_path ===> ',output_path)
        print('---------------END Initialisation-------------')
        return file_path
    
    def start_training(self):
        print('------------Start training [ '+str(self.train)+' ]--------')
        global output_path
        self.train=True
        self.exp_D.configure(text="ACTUAL EXPERIMENT NUMBER  : "+str(experimentInedx))
        var=output_path.replace("output.txt","")
        self.checkpoints_D.configure(text="CHECKPOINTS: "+str(var))
        
        #--- create output and all folders----
        IN=output_path
        #----------------------------------
        with open(IN,'a') as file:
            file.write('The training is start ......'+'\n')       
        self.update_gui_with_file_content()
        process = subprocess.Popen([
                "python",
                "C:\\bilel\\FastFlow\\trainFile\\main_test_all_v2.py",
                "--train"
            ])
        self.inspection_processes.append(process)
        self.initialize_real_time_plotter(output_path)
        
        print('------------end start training ------------------')
   
    def update_gui_with_file_content(self):
        global output_path
        try:
            # print('in update_gui_with_file_content :',output_path)
            with open(output_path, 'r') as file:
                content = file.read()
                self.training_output.delete(1.0, tk.END)  # Clear existing content
                self.training_output.insert(tk.END, content)  # Insert new content
        except IOError:
            print("Error reading output.txt")

        # Schedule this method to be called again after 1000 ms (1 second)
        if self.train:
            self.after(1000, self.update_gui_with_file_content) 

    def check_action(self):
        # Read values from entry widgets
        batch_size = self.entry1.get()
        num_epochs = self.entry2.get()
        input_size = self.entry3.get()

        # Optionally, update these values in the configuration file
        self.update_config_file(batch_size, num_epochs,input_size)

    def update_config_file(self, batch_size, num_epochs,input_size):
        try:
            # File path
            config_path = self.constantPath

            # Read the file
            with open(config_path, 'r') as file:
                lines = file.readlines()

            # Update the lines
            with open(config_path, 'w') as file:
                for line in lines:
                    if line.startswith('BATCH_SIZE'):
                        file.write(f"BATCH_SIZE = {batch_size}\n")
                        self.batch_size_label.configure(text="BATCH_SIZE : "+str(batch_size))
                    elif line.startswith('NUM_EPOCHS'):
                        file.write(f"NUM_EPOCHS = {num_epochs}\n")
                        self.num_epochs_label_D.configure(text="NUM_EPOCHS : "+str(num_epochs))
                    elif line.startswith('INPUT_SIZE'):
                        file.write(f"INPUT_SIZE = {input_size}\n")
                        self.IMAGES_SIZE_D.configure(text="TRAINIG_INPUT_SIZE : "+str(input_size)+" x "+str(input_size))
                    else:
                        file.write(line)

            print(f"Configuration updated: BATCH_SIZE={batch_size}, NUM_EPOCHS={num_epochs}, INPUT_SIZE={input_size}")

        except ValueError:
            print("Invalid input: BATCH_SIZE and NUM_EPOCHS must be integers.")
        except Exception as e:
            print(f"Error updating configuration file: {e}")

    def stop_training(self, index=None):
        global output_path
        print('in stop training function')
        
        if index is not None and 0 <= index < len(self.inspection_processes):
            # Terminate a specific subprocess by index
            self.inspection_processes[index].terminate()
            # print(f"Inspection {index} stopped")
            self.train=False
            consts.TRAIN_STATUS=False
             # Insert new content
        else:
            # Terminate all subprocesses
            for process in self.inspection_processes:
                process.terminate()
            if self.train:
                with open(output_path,'a') as file:
                    file.write('The training is STOP ....!'+'\n')
                    self.train=False
                    consts.TRAIN_STATUS=False

    def update_value(self,file_path, constant_name, new_value):
    # Read the file
        with open(file_path, 'r',encoding='utf-8') as file:
            lines = file.readlines()

        # Modify the desired constant
        for i, line in enumerate(lines):
            if line.startswith(constant_name):
                lines[i] = f"{constant_name} = {repr(new_value)}\n"
                break

        # Write back to the file
        with open(file_path, 'w',encoding='utf-8') as file:
            file.writelines(lines)  

    def Clear_train(self):
        print('**************** Clear_train *********************')
        global output_path
        folder_path= consts.CHECKPOINT_DIR
        # Check if the folder exists
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return
        response = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete the experiments folder?")
        if response:
            try:
                shutil.rmtree(folder_path)
                print(f"Folder '{folder_path}' has been deleted.")
                self.Restart_train(True)
                output_path=consts.OUTPUT_FILE_PATH
            except Exception as e:
                print(f"Error occurred while deleting folder: {e}")
        else:
            print("Deletion cancelled.")

    def Restart_train(self,perform_action=False):
        print('------------ restart -----------------')
        global experimentInedx
        global output_path
        #update the experiment index
        if perform_action:
            experimentInedx=1
        else:
            experimentInedx=experimentInedx+1
    
        self.exp_D.configure(text="ACTUAL EXPERIMENT NUMBER  : "+str(experimentInedx))
        self.update_value(self.constantPath,'SCORE_ALL',"_")
        # self.acc_v.configure(text='_')
        #creat the checkpoint_dir path and experiment folder
        checkpoint_path = os.path.join(
            consts.CHECKPOINT_DIR, "exp%d" % experimentInedx  )
        checkpoint_path= os.path.abspath(checkpoint_path)
        # update the CHECKPOINT_PATH
        self.update_constant_file('CHECKPOINT_PATH',checkpoint_path)
        self.checkpoints_D.configure(text="CHECKPOINTS: "+checkpoint_path)
        #creat  experiment folder
        os.makedirs(checkpoint_path, exist_ok=True)
        print('creat file : ',checkpoint_path)
        #creat the path of output.txt
        file_path= os.path.join(checkpoint_path, 'output.txt')
        #update the OUTPUT_FILE_PATH
        self.update_constant_file('OUTPUT_FILE_PATH',file_path)
        self.reload_constants()
        output_path=file_path
        #create output.txt file in experimwnt folder
        if not os.path.exists(file_path):
        # Create the file output.txt
            with open(file_path, 'w') as file:
                file.write("")
        
        self.update_gui_with_file_content()
        self.initialize_real_time_plotter(output_path)
        print('-----------finish restart-------------')


    #FRAME TEST FUNCTION
    def start_test_All(self):
        print('*****************we are in START MULTIPLE TEST **************')
        # Change cursor to 'wait'
        self.content_frame.configure(cursor="wait")

        # Start the test in a new thread
        test_thread = threading.Thread(target=self.start_test_All_images)
        test_thread.start()
    
    def updateImages(self):
        self.update_constant_file('IMAGE_TEST_PATH',os.path.join(self.current_dir, "inputE.png"))
        self.update_constant_file('IMAGE_PREDECTION_PATH',os.path.join(self.current_dir, "waiting.png"))
        self.reload_constants()
        fixed_size = (512, 512)  
        inImageTest1=consts.IMAGE_TEST_PATH
        image = Image.open(inImageTest1)  # Replace with your image path
        image = image.resize(fixed_size,  Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.input_test_label.config(image=photo)
        self.input_test_label.image = photo

        inImageTest2=consts.IMAGE_PREDECTION_PATH
        
        image = Image.open(inImageTest2)  # Replace with your image path
        image = image.resize(fixed_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.predection_label.config(image=photo)
        self.predection_label.image = photo

    def start_test_All_images(self):
        print('//// we are in all images test  section /////')
        self.reload_constants()
        self.updateImages()
        weight_path = getattr(consts, 'WEIGHT_PATH', None)
        if not weight_path or weight_path == '_' or not weight_path.endswith('.pt'):
            messagebox.showerror("Invalid Weight Path", "Please set a valid weight path (.pt file) before starting the test.")
            self.content_frame.configure(cursor="")  # Reset cursor if no valid image
            return  # Stop the function if the weight path is not valid
        
        
        response = messagebox.askyesno("Dataset Path Required", "The dataset path is not set. Do you want to select a dataset path now?")
        if response:
            if self.select_test_all_path() :

                self.update_value(self.constantPath,'SCORE_ALL','_')
                    
        # Start main.py script as a subprocess and append it to the list
                process = subprocess.Popen([
                    "python",
                    "C:\\bilel\\FastFlow\\trainFile\\main_test_all_v2.py",
                    "--eval_All"
                ])
                self.inspection_processes.append(process)
                messages = "Starting the MULTIPLE_TEST_PATH.\nPlease wait until the process is completed."
                messagebox.showinfo("MULTIPLE_TEST_PATH", messages)
                    # Wait for the process to complete
                process.wait()
                print('SCORE_ALL---> :',consts.SCORE_ALL)
                self.reload_constants()
                self.content_frame.configure(cursor="")
                percentage_value = self.decimal_to_percentage(consts.SCORE_ALL)
                messages = F"SCORE IS : {percentage_value}"
                messagebox.showinfo("Accurecy SCORE ", messages)
            else:
                self.content_frame.configure(cursor="")
                return  # Stop the function if the user does not want to set the path
    
    def decimal_to_percentage(self,decimal_number):
        decimal_number=float(decimal_number)
        percentage = decimal_number * 100
        print('percentage--->',percentage)
        return f'{percentage:.0f}%'          
#calc threshold
    def start_th(self):
        print('we are in start_threshold **************')
        # Change cursor to 'wait'
        self.content_frame.configure(cursor="wait")

        # Start the test in a new thread
        test_thread = threading.Thread(target=self.start_calc_th)
        test_thread.start()
    
    def start_calc_th(self):
        self.reload_constants()
        print('//// we are in calculate trushold section  section /////')
        weight_path = getattr(consts, 'WEIGHT_PATH', None)
        print('befor==========================> ',consts.WEIGHT_PATH)
        if not weight_path or weight_path == '_' or not weight_path.endswith('.pt'):
            print('after==========================> ',consts.WEIGHT_PATH)
            messagebox.showerror("Invalid Weight Path", "Please set a valid weight path (.pt file) before starting the test.")
            self.content_frame.configure(cursor="")  # Reset cursor if no valid image
            return  # Stop the function if the weight path is not valid
        if not hasattr(consts, 'DATASET_PATH') or consts.DATASET_PATH == '_':
            response = messagebox.askyesno("Dataset Path Required", "The dataset path is not set. Do you want to select a dataset path now?")
            if response:
                self.select_dataset_path_good()
            else:
                self.content_frame.configure(cursor="")
                return  # Stop the function if the user does not want to set the path
            
        #delete old result of score 
        # self.update_value(self.constantPath,'SCORE_ALL','_')
        if (self.data_selected):
            # Start main.py script as a subprocess and append it to the list
            process = subprocess.Popen([
                "python",
                "C:\\bilel\\FastFlow\\trainFile\\main_test_all_v2.py",
                "--TH"
            ])
            self.inspection_processes.append(process)
            messages = "Starting threshold calculation.\nPlease wait until the process is completed."
            messagebox.showinfo("CALCULATE THRESHOLD", messages)
                # Wait for the process to complete
            process.wait()
            # print('SCORE_ALL---> :',consts.SCORE_ALL)
            self.reload_constants()
            self.TH_entry.delete(0,'end')
            self.TH_entry.insert(0, consts.THRESHOLD)
            self.content_frame.configure(cursor="")
    def caputre_image(self):
        print('Button capture image...............') 
        self.capture_window = CameraCapture(self)

    
    def load_image(self):
        # Open a file dialog to select the image
        image_path = filedialog.askopenfilename()
        global inImageTest 
        # Check if a file was selected
        if image_path:
            # Check if the file is an image
            if imghdr.what(image_path) is not None:
                self.update_constant_file('IMAGE_TEST_PATH', image_path)
                fixed_size = (512, 512)  
                image = Image.open(image_path)
                image = image.resize(fixed_size,  Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.input_test_label.configure(image=photo)
                self.input_test_label.image = photo
                self.update_value(self.constantPath,'PREDECTION_RESUILT','No result...')
                self.input_test_label.config(text=f"Score: ---")  
                messagebox.showinfo("Success", "Test image successfully loaded.")
            else:
                # Display a popup message if the file is not an image
                messagebox.showerror("Invalid File", "The selected file is not an image.")
        else:
            # Display a popup message if no file is selected
            messagebox.showinfo("No File Selected", "Please select a file.")
 

    def start_test(self):
        # Change cursor to 'wait'
        self.content_frame.configure(cursor="wait")

        # Start the test in a new thread
        test_thread = threading.Thread(target=self.start_test_single)
        test_thread.start()

    def start_test_single(self):
        
        print('//// we are in test section /////')
        self.reload_constants()
        weight_path = getattr(consts, 'WEIGHT_PATH', None)
        if not weight_path or weight_path == '_' or not weight_path.endswith('.pt'):
            messagebox.showerror("Invalid Weight Path", "Please set a valid weight path (.pt file) before starting the test.")
            self.content_frame.configure(cursor="")  # Reset cursor if no valid image
            return  # Stop the function if the weight path is not valid
        
        test_image_path= consts.IMAGE_TEST_PATH
        print(test_image_path)

        # Check if the test image is the initial image (inputE.png)
        if not test_image_path or os.path.basename(test_image_path) == 'inputE.png':
            messagebox.showerror("Invalid Image", "There is no new test image loaded. Please load a test image first.")
            self.content_frame.configure(cursor="")  # Reset cursor
            return
         # Check if THRESHOLD is set in consts
        threshold = getattr(consts, 'THRESHOLD', None)
        if threshold is None:
            messagebox.showerror("Missing Threshold", "Please set a threshold with Starting start_test_all for threshold initialization")
            self.content_frame.configure(cursor="")  # Reset cursor
            return
        self.input_test_label.config(text=f"Score: ---")  
        if test_image_path and imghdr.what(test_image_path)is not None:
        # Start main.py script as a subprocess and append it to the list
            process = subprocess.Popen([
                "python",
                "C:\\bilel\\FastFlow\\trainFile\\main_test_all_v2.py",
                "--eval_One","--image_path",test_image_path
            ])
            self.inspection_processes.append(process)
            
            messagebox.showinfo("Test Starting", "The test is starting. Please wait...")
            # Wait for the process to complete
            process.wait()
            self .reload_constants()
            self.check_and_display_score()
            # Check the result and update the image
            self.check_result()
        else:
            print('There is no image test input.\n  Please load image test first')
            messagebox.showerror("Invalid File", "There is no image test input.")
            self.content_frame.configure(cursor="")  # Reset cursor if no valid image

    def folder_contains_images(self, folder):
        # Common image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        # Check if folder contains files with these extensions
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                return True

        # No images found, show popup
        messagebox.showerror("No Images Found", f"No image files found in the folder: {folder}")
        return False
    def check_result(self):
        print("check result------------------>", consts.PREDECTION_RESUILT )
        # Assuming PREDECTION_RESULT is set in consts
        if consts.PREDECTION_RESUILT == "good":
            
            image_path = os.path.join(self.current_dir, "positive.jpg")
            print('good path',image_path)
            # Load and set the new image
            self.update_image(image_path)

        if consts.PREDECTION_RESUILT == "bad":
            image_path = os.path.join(self.current_dir, "negative.jpg")
            print('bad path',image_path)
            # Load and set the new image
            self.update_image(image_path)

        
        # Reset cursor
        self.content_frame.configure(cursor="")


    def update_image(self, image_path):
        # Update the image in the label
        image = Image.open(image_path)
        image = image.resize((512, 512), Image.LANCZOS)  # Update size as needed
        photo = ImageTk.PhotoImage(image)

        self.predection_label.config(image=photo)
        self.predection_label.image = photo
if __name__ == "__main__":
    app = App()
    app.mainloop()
