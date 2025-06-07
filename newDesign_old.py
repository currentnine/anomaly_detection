# from main_test_all_v2 import hi
import tkinter as tk
import customtkinter
from realtime_plotting1 import RealTimePlotter
from tkinter import ttk, filedialog, messagebox
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
        # self.bad_data_selected = False
        # Sidebar
        self.initialisation_folder()
        # output_path=consts.OUTPUT_FILE_PATH
        output_path = self.initialisation_folder()
        
        sidebar = customtkinter.CTkFrame(self, width=200)
        sidebar.pack(side="left", fill="y", padx=5, pady=5)

        # Train button
        train_button = customtkinter.CTkButton(sidebar, text="Train", command=lambda: self.on_tab_selected("train"))
        train_button.pack(pady=10, fill="x")

        # Test button
        test_button = customtkinter.CTkButton(sidebar, text="Test", command=lambda: self.on_tab_selected("test"))
        test_button.pack(pady=10, fill="x")

        # Bottom frame for exit button
        bottom_frame = customtkinter.CTkFrame(sidebar)
        bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Exit button
        exit_button = customtkinter.CTkButton(bottom_frame, fg_color="red", text="Exit", command=self.quit_application)
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
        self.button_start = customtkinter.CTkButton(tab1, text="Start",command=self.start_training,  state=tk.DISABLED)
        self.button_start.grid(row=0, column=0, padx=10, pady=10)

        self.button_stop = customtkinter.CTkButton(tab1, text="Stop",command=self.stop_training, state=tk.DISABLED)
        self.button_stop.grid(row=0, column=1, padx=10, pady=10)

        self.button_good_dataset = customtkinter.CTkButton(tab1, text="Select Dataset file", command=self.select_dataset_path)
        self.button_good_dataset.grid(row=0, column=2, padx=10, pady=10)

        self.button_restart = customtkinter.CTkButton(tab1, text="Restart",command=self.Restart_train,  state=tk.DISABLED)
        self.button_restart.grid(row=0, column=4, padx=10, pady=10)

        self.button_clear = customtkinter.CTkButton(tab1, text="Clear & Reinitialize",command=self.Clear_train)
        self.button_clear.grid(row=0, column=6, padx=10, pady=10)
        tab1.grid_rowconfigure(1, weight=1)
        tab1.grid_rowconfigure(2, weight=1)


        self.update_start_button_state()

        # Assigning weights to columns
        weights = [1, 1, 1, 1, 1, 1]  # Total weight = 6
        for col, weight in enumerate(weights):
            tab1.grid_columnconfigure(col, weight=weight)

        # Place CTkTextbox in the first column and full vertical space from the second row
        self.training_output = customtkinter.CTkTextbox(tab1)
        self.training_output.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        self.update_gui_with_file_content()

        #-------------"Grid 1" - Configuration Frame----------
        config_frame = ttk.LabelFrame(tab1, text="Configuration")
        # Label and Entry for BATCH_SIZE
        batch_size_label = ttk.Label(config_frame, text="BATCH_SIZE")
        batch_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry1 = ttk.Entry(config_frame, width=20)
        self.entry1.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.entry1.insert(0, consts.BATCH_SIZE)  # Set BATCH_SIZE value

        # Label and Entry for NUM_EPOCHS
        num_epochs_label = ttk.Label(config_frame, text="NUM_EPOCHS")
        num_epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry2 = ttk.Entry(config_frame, width=20)
        self.entry2.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry2.insert(0, consts.NUM_EPOCHS)  # Set NUM_EPOCHS value
        # Label and Entry for NUM_EPOCHS
        input_size_label = ttk.Label(config_frame, text="TRAINING_INPUT_SIZE")
        input_size_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry3 = ttk.Entry(config_frame, width=20)
        self.entry3.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.entry3.insert(0, consts.INPUT_SIZE)  # Set NUM_EPOCHS value
        # Add a check button at the bottom
        check_button = ttk.Button(config_frame, text="Confirm", command=self.check_action)
        check_button.grid(row=3, columnspan=2, padx=5, pady=10)

        # Place the LabelFrame within the grid
        config_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nw")
        config_frame.grid(row=1, column=1, columnspan=2, sticky="nw", padx=10, pady=10)

        #-------------"Grid 2"----------
        config_frameG2 = ttk.LabelFrame(tab1, text="Display training parameter")
        #BATCH_SIZE
        
        batch_size_label = ttk.Label(config_frameG2, text="BATCH_SIZE : ")
        batch_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.batch_size_labelValue = ttk.Label(config_frameG2, text=consts.BATCH_SIZE)
        self.batch_size_labelValue.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        #NUM_EPOCHS
        num_epochs_label_D = ttk.Label(config_frameG2, text="NUM_EPOCHS : ")
        num_epochs_label_D.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_epochs_label_Value = ttk.Label(config_frameG2, text=consts.NUM_EPOCHS)
        self.num_epochs_label_Value.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        # LEARNINGN RATE
        LR_Label_D = ttk.Label(config_frameG2, text="LEARNINGN RATE : ")
        LR_Label_D.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.LR_Label_Value = ttk.Label(config_frameG2, text=consts.LR)
        self.LR_Label_Value.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        # IMAGES SIZE
        IMAGES_SIZE_D = ttk.Label(config_frameG2, text="TRAINIG_INPUT_SIZE : ")
        IMAGES_SIZE_D.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        sizeInput=str(consts.INPUT_SIZE)
        self.IMAGES_SIZE_D_Value = ttk.Label(config_frameG2, text=sizeInput+' x '+sizeInput)
        self.IMAGES_SIZE_D_Value.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        # DATASET GOOD
        DATASET_GOOD = ttk.Label(config_frameG2, text=" DATASET PATH  : ")
        DATASET_GOOD.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.DATASET_GOOD_path = ttk.Label(config_frameG2, text='_')
        self.DATASET_GOOD_path.grid(row=4, column=1, padx=5, pady=5, sticky="w")
 
        # fastflow_experiment_checkpoints
        checkpoints_D = ttk.Label(config_frameG2, text="Checkpoints  : ")
        checkpoints_D.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.checkpoints_path = ttk.Label(config_frameG2, text="_")
        self.checkpoints_path.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        #experiment 
        
        experimentInedx=len(os.listdir(consts.CHECKPOINT_DIR))
        exp_D = ttk.Label(config_frameG2, text="Actual experiment number  : ")
        exp_D.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.exp_V = ttk.Label(config_frameG2, text=experimentInedx)
        self.exp_V.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        
        # self.get_last_experiment(checkpoint_path)
        #----
        config_frameG2.grid(row=1, column=1, padx=10, pady=10, sticky="nw")
        config_frameG2.grid(row=1, column=3, columnspan=3, sticky="nw", padx=10, pady=10)

        #-------------"Grid 3"----------
        self.real_time_plotter = None
        self.initialize_real_time_plotter(output_path)
        

         #configuration tab2
        self.CAPTURE_BTN = customtkinter.CTkButton(tab2, text="CAPTURE",command=self.caputre_image)
        self.CAPTURE_BTN.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), sticky="w")
        self.LOAD_IMAGE_BTN = customtkinter.CTkButton(tab2, text="LOAD IMAGE",command=self.load_image)
        self.LOAD_IMAGE_BTN.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="w")
        self.LOAD_W_1TEST_BTN = customtkinter.CTkButton(tab2, text="LOAD WIEGHT",command=self.laod_wieght)
        self.LOAD_W_1TEST_BTN.grid(row=2, column=2, padx=(10, 10), pady=(10, 10), sticky="w")
        self.START_SINGLE_TEST_BTN = customtkinter.CTkButton(tab2, text="START SINGLE TEST ",command=self.start_test)
        self.START_SINGLE_TEST_BTN.grid(row=2, column=3, padx=(10, 10), pady=(10, 10), sticky="w")

        config_frame4 = ttk.LabelFrame(tab2, text="START ALL TEST & CALCULATE THRUSHOLD")
        # Label and Entry for BATCH_SIZE
        check_button = ttk.Button(config_frame4, text="Select weight ", command=self.laod_wieght)
        check_button.grid(row=0, column=1, padx=5, pady=10)
        check_button = ttk.Button(config_frame4, text="Start test all images", command=self.start_test_All)
        check_button.grid(row=2, column=1, padx=5, pady=10)
        acc = ttk.Label(config_frame4, text="accuracy_score : ")
        acc.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.acc_v = ttk.Label(config_frame4, text='_')
        self.acc_v.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        self.check_for_update()

        #Weight selected
        weight_L=ttk.Label(config_frame4, text="Weight_Selected : ")
        weight_L.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.weight_v = ttk.Label(config_frame4, text=weight_path)
        self.weight_v.grid(row=6, column=2, padx=5, pady=5, sticky="w")

        # Place the LabelFrame within the grid
        config_frame4.grid(row=3, column=0, columnspan=2, sticky="nw", padx=10, pady=10)

        #display Image and predection----------------
        fixed_size = (512, 512)  
        background_color = self.cget("background")
        self.input_test_label = ttk.Label(tab2, text="Input Image",font=("Arial", 12, "bold"), compound="bottom",foreground="white", background=background_color)
        inImageTest1=consts.IMAGE_TEST_PATH
        print('inImageTest1==>',inImageTest1)
        image = Image.open(inImageTest1)  # Replace with your image path
        image = image.resize(fixed_size,  Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.input_test_label.config(image=photo)
        self.input_test_label.image = photo
        self.input_test_label.grid(row=4, column=0, padx=0, pady=10, sticky="nsew")
        self.input_test_label.bind("<Button-1>", lambda event: self.load_image())
        self.input_test_label.bind("<Enter>", lambda event: self.input_test_label.config(cursor="hand2"))
        self.input_test_label.bind("<Leave>", lambda event: self.input_test_label.config(cursor=""))

        inImageTest2=consts.IMAGE_PREDECTION_PATH
        self.predection_label = ttk.Label(tab2, text="Predection",font=("Arial", 12, "bold"), compound="bottom",foreground="white", background=background_color)
        self.predection_label.grid(row=4, column=4, padx=0, pady=10, sticky="nsew")
        image = Image.open(inImageTest2)  # Replace with your image path
        image = image.resize(fixed_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        # Set the image in the label
        self.predection_label.config(image=photo)
        self.predection_label.image = photo

        
        
    def reload_constants(self):
        importlib.reload(consts)

    def check_for_update(self):
        self.reload_constants()
        current_value = consts.SCORE_ALL

        # Check if acc_v is initialized
        if hasattr(self, 'acc_v'):
            self.acc_v.config(text=str(current_value))
        else:
            print("acc_v is not initialized")
        self.after(1000, self.check_for_update)
    
    def laod_wieght1(self):
        global weight_path
        weight_path = filedialog.askopenfilename()
        self.update_constant_file('WEIGHT_PATH', weight_path)
        self.weight_v.configure(text=weight_path)
        print(weight_path)
        
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
    def initialize_real_time_plotter(self,file_path):
        # print("plot initialize_real_time_plotter output : ",file_path)
        # print("plot initialize_real_time_plotter output : ",consts.OUTPUT_FILE_PATH)
        # Remove the existing RealTimePlotter instance
        if self.real_time_plotter is not None:
            self.real_time_plotter.grid_forget()  # This removes the widget from the layout
        # Create a new RealTimePlotter instance
        self.real_time_plotter = RealTimePlotter(self.train_frame ,file_path)  # Replace with appropriate arguments

        # Place the new instance in the grid
        self.real_time_plotter.grid(row=2, column=1, columnspan=4, sticky="nsew", padx=10, pady=10)

    def get_last_experiment(self,folder_path):
        folder_abspath= folder_path
        # Check if the given path is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a directory.")
            os.makedirs(folder_path, exist_ok=True)
            self.checkpoints_path.configure(text=folder_abspath)
            return 0
        # List all items in the directory
        all_items = os.listdir(folder_path)
        # Filter out items that are not directories or don't follow the 'expX' pattern
        exp_directories = [item for item in all_items if item.startswith('exp') and os.path.isdir(os.path.join(folder_path, item))]
        # Sort the directories by the numerical part of their names
        exp_directories.sort(key=lambda x: int(x[3:]))  # Assuming the format is 'exp' followed by a number
        # Return the last experiment directory, if any
        if exp_directories:
            i=int((exp_directories[-1]).replace("exp",""))
            self.exp_V.configure(text=i)
            self.checkpoints_path.configure(text=folder_abspath)
            return os.path.join(folder_path, exp_directories[-1])
        else:
            self.checkpoints_path.configure(text=folder_abspath)
            print("No experiment directories found ---> creat new one .")
            self.exp_V.configure(text='0')
            return 0

    def select_dataset_path(self):
        # Open a dialog to select a folder
        folder_selected = filedialog.askdirectory()
        print(folder_selected)
        if folder_selected:
            # Check if 'test' and 'train' folders exist in the selected directory
            required_folders = ['test', 'train']
            if all(os.path.isdir(os.path.join(folder_selected, folder)) for folder in required_folders):
                # Update the constant in constant.py
                if self.update_constant_file('DATASET_PATH', folder_selected):
                    self.DATASET_GOOD_path.configure(text=folder_selected)
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

            print(f"Old value for '{constant_name}': {old_value}")

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

            print(f"New value for '{constant_name}' in file: {new_value_in_file}")

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
        #-----
        # if hasattr(self, 'exp'):
        #     print("self.exp is initialized.")
        #     self.exp_V.configure(text=id)
        # else:
        #     print("self.exp is not initialized.")
        #-----
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
        print('---------------END Initialisation-------------')
        return file_path
    
    def start_training(self):
        print('------------Start training [ '+str(self.train)+' ]--------')
        global output_path
        print('experimentInedx : ',experimentInedx)
        self.train=True
        self.exp_V.configure(text=experimentInedx)
        #self.initialisation_folder()
        #self.get_last_experiment(consts.CHECKPOINT_PATH)
        var=output_path.replace("output.txt","")
        print('update with consts.CHECKPOINT_PATH =>output_path: ',var)
        self.checkpoints_path.configure(text=var)
        
        #--- create output and all folders----
        IN=output_path
        print('THIS IS :=====> : ',IN)
        #----------------------------------
        with open(IN,'a') as file:
            file.write('The training is start ......'+'\n')
        
        self.update_gui_with_file_content()
        
        # Start main.py script as a subprocess and append it to the list
        # process = subprocess.Popen(["python", "C:\\bilel\\FastFlow\\trainFile\\main.py"])
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
                        self.batch_size_labelValue.configure(text=batch_size)
                    elif line.startswith('NUM_EPOCHS'):
                        file.write(f"NUM_EPOCHS = {num_epochs}\n")
                        self.num_epochs_label_Value.configure(text=num_epochs)
                    elif line.startswith('INPUT_SIZE'):
                        file.write(f"INPUT_SIZE = {input_size}\n")
                        self.IMAGES_SIZE_D_Value.configure(text=str(input_size)+" x "+str(input_size))
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
            print(f"Inspection {index} stopped")
            self.train=False
            consts.TRAIN_STATUS=False
             # Insert new content
        else:
            # Terminate all subprocesses
            for process in self.inspection_processes:
                process.terminate()
            # self.inspection_processes.clear()
            # content ='The training is STOP ....!'
            #self.training_output.insert(tk.END, content) 
            if self.train:
                with open(output_path,'a') as file:
                    file.write('The training is STOP ....!'+'\n')
                    self.train=False
                    consts.TRAIN_STATUS=False
      
          #print(content) 
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
        global output_path
        print('**************** Clear_train *********************')
        
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
                print('Clear and deleeeeeee : output',output_path)
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
            experimentInedx=0
            print("deleteeeeeeeeeeeeeeeeeee11eeeeeeeee",experimentInedx)
        else:
            experimentInedx=experimentInedx+1

            print("deleteeeeeeeeeeeeeeeeee22eeeeeeeeee",experimentInedx)
        self.exp_V.configure(text=experimentInedx)
        self.update_value(self.constantPath,'SCORE_ALL',"_")
        self.acc_v.configure(text='_')
        #creat the checkpoint_dir path and experiment folder
        checkpoint_path = os.path.join(
            consts.CHECKPOINT_DIR, "exp%d" % experimentInedx  )
        checkpoint_path= os.path.abspath(checkpoint_path)
        # update the CHECKPOINT_PATH
        self.update_constant_file('CHECKPOINT_PATH',checkpoint_path)
        print('update checkpoitn path in constants : ',checkpoint_path)
        self.checkpoints_path.configure(text=checkpoint_path)
        #creat  experiment folder
        os.makedirs(checkpoint_path, exist_ok=True)
        print('creat file : ',checkpoint_path)
        #creat the path of output.txt
        file_path= os.path.join(checkpoint_path, 'output.txt')
        #update the OUTPUT_FILE_PATH
        self.update_constant_file('OUTPUT_FILE_PATH',file_path)
        output_path=file_path
        print('output_path====:',output_path )
        #create output.txt file in experimwnt folder
        if not os.path.exists(file_path):
        # Create the file output.txt
            with open(file_path, 'w') as file:
                file.write("")
        
        self.update_gui_with_file_content()
        self.initialize_real_time_plotter(output_path)
        print('-----------finish restart-------------')

    def clear_output_file(self):
        global output_path
        try:
            with open(output_path, 'w') as file:
                file.write("")  # Write an empty string to clear the file content
            print('Output file content cleared')
            self.train=False
            self.stop_training()
            self.update_gui_with_file_content()
        except IOError:
            print("Error clearing output.txt")   
    def clear_output_file2(self):
        try:
            # Clear the content of the 'output.txt' file
            with open(consts.OUTPUT_FILE_PATH, 'w') as file:
                file.write("")  # Write an empty string to clear the file content
            
            # Clear the content of the Text widget
            self.training_output.delete(1.0, tk.END)
            print('Output file content and Text widget content cleared')
        except IOError:
            print("Error clearing output.txt")
    def clear_text_widget(self):
        # Clear the content of the Text widget
        self.training_output.delete(1.0, tk.END)
        print('Text widget content cleared')


    #FRAME TEST FUNCTION
    def start_test_All(self):
        #select_dataset_path
        print('we are in start_test_All **************')
        # Change cursor to 'wait'
        self.content_frame.configure(cursor="wait")

        # Start the test in a new thread
        test_thread = threading.Thread(target=self.start_test_All_images)
        test_thread.start()
    
    def start_test_All_images(self):
        print('//// we are in all images test  section /////')
        if not hasattr(consts, 'DATASET_PATH') or consts.DATASET_PATH == '_':
            response = messagebox.askyesno("Dataset Path Required", "The dataset path is not set. Do you want to select a dataset path now?")
            if response:
                self.select_dataset_path()
            else:
                self.content_frame.configure(cursor="")
                return  # Stop the function if the user does not want to set the path
            
        #delete old result of score 
        self.update_value(self.constantPath,'SCORE_ALL','_')
        if (self.data_selected):
            # Start main.py script as a subprocess and append it to the list
            process = subprocess.Popen([
                "python",
                "C:\\bilel\\FastFlow\\trainFile\\main_test_all_v2.py",
                "--eval_All"
            ])
            self.inspection_processes.append(process)
            messages = "Starting the test for all images, including threshold calculation.\nPlease wait until the process is completed."
            messagebox.showinfo("Test Starting for all Images ", messages)
                # Wait for the process to complete
            process.wait()
            self.content_frame.configure(cursor="")
    def caputre_image(self):
        print('Button capture image...............') 

    
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

            # Check the result and update the image
            self.check_result()
        else:
            print('There is no image test input.\n  Please load image test first')
            messagebox.showerror("Invalid File", "There is no image test input.")
            self.content_frame.configure(cursor="")  # Reset cursor if no valid image

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
