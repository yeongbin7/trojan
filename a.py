from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import glob
import os
from sklearn.decomposition import PCA

tk = Tk()
tk.title("Trojan!!!")
tk.geometry("600x400+100+100")
tk.resizable(True,True)
label = Label(tk, text="Trojan program")
label.pack(side=TOP, padx=10, pady=10)
number = 0
processing_file_path = ""
preprocessing_folder_path = ""
preprocessing_button1 = Button()
preprocessing_button2 = Button()
preprocessing_button3 = Button()

def event():
    global number
    number += 1
    button['text'] = '버튼 누른횟수: ' + str(number)
    
def file_load_event():
    global processing_file_path
    processing_file_path = filedialog.askdirectory()
    print(processing_file_path)
    name_list = os.listdir(processing_file_path)
    path_name = []
    for i in range(len(name_list)):
        path_name.append(processing_file_path + name_list[i])
    print(path_name)
    
def make_preprocessing_folder():
    global preprocessing_button1
    global preprocessing_folder_path
    processing_dir = filedialog.askdirectory()
    preprocessing_folder_path = processing_dir +  "/preprocessing"
    # folder 없어도 넘어감
    os.makedirs(preprocessing_folder_path, exist_ok=True)
    
    messagebox.showinfo("Message", "Success for making preprocessing folder!!")
    preprocessing_button1['text'] = "Success for making folder"

def make_header_trusted_data():

    print(preprocessing_folder_path)
    trust_dir = filedialog.askdirectory() + "/"
    trust_name_list = os.listdir(trust_dir)
    path_trust_name = []
    for i in range(len(trust_name_list)):
        path_trust_name.append(trust_dir + trust_name_list[i])
    
    for i in range(len(path_trust_name)):
        sub_list = []
        file_free = glob.glob(os.path)
        for f in file_free:
            sub_list.append(pd.read_csv(f))
        data_free_raw = pd.concat(sub_list[:], axis=1)
        data_free_amp = (data_free_raw.drop(columns='Time', axis=1)).T
        output_path = path_trust_name[i] + '.h5'
        data_free_amp.to_hdf(output_path, 'a')

    messagebox.showinfo("Message", "Success for making header file from trusted data")
    preprocessing_button1['text'] = "Success for making header file from trusted data"

def make_header_trojan_data():

    print(preprocessing_folder_path)
    trojan_dir = filedialog.askdirectory() + "/"
    trojan_name_list = os.listdir(trojan_dir)
    path_trojan_name = []
    for i in range(len(trojan_name_list)):
        path_trojan_name.append(trojan_dir + trojan_name_list[i])
    for i in range(len(path_trojan_name)):
        sub_list = []
        file_free = glob.glob(os.path)
        for f in file_free:
            sub_list.append(pd.read_csv(f))
        data_free_raw = pd.concat(sub_list[:], axis=1)
        data_free_amp = (data_free_raw.drop(columns='Time', axis=1)).T
        output_path = path_trojan_name[i] + '.h5'
        data_free_amp.to_hdf(output_path, 'a')

    messagebox.showinfo("Message", "Success for making header file from trojan data")
    preprocessing_button1['text'] = "Success for making header file from trojan data"

def Preprocess_trusted_data():
    trust_file_list = []
    trust_file_dir = glob.glob(os.path.join(filedialog.askdirectory(), "*.h5"))
    for i in trust_file_dir:
        trust_file_list.append(pd.read_hdf(i))
    n_pca = 1000
    

    messagebox.showinfo("Message", "Success for preprocessing trusted data")
    preprocessing_button1['text'] = "Success for preprocessing trusted data"

def Preprocess_trojan_data():
    trojan_file_list = []
    trojan_file_dir = glob.glob(os.path.join(filedialog.askdirectory(), "*.h5"))

    for i in trojan_file_dir:
        trojan_file_list.append(pd.read_hdf(i))
    n_pca = 1000
    # TODO: data cutting
    for i in range(len(trojan_file_list)):
        troj_mean_data = 

    messagebox.showinfo("Message", "Success for preprocessing trojan data")
    preprocessing_button1['text'] = "Success for preprocessing trojan data"

def preprocessing():
    global preprocessing_button1, preprocessing_button2, preprocessing_button3
    newWindow = Toplevel(tk)
    newWindow.geometry("400x200+100+100")
    labelExample = Label(newWindow, text="Preprocessing")
    preprocessing_button1 = Button(newWindow, text="1. Make preprocessing folder", command=make_preprocessing_folder)
    preprocessing_button1.pack()
    preprocessing_button2 = Button(newWindow, text="2. Make header as trusted csv data", command=make_header_trusted_data)
    preprocessing_button2.pack()
    preprocessing_button3 = Button(newWindow, text="3. Preprocess trusted data", command=make_header_trojan_data)
    preprocessing_button3.pack()
    preprocessing_button4 = Button(newWindow, text="4. Make header as trojan csv data", command=make_header_trojan_data)
    preprocessing_button4.pack()
    preprocessing_button5 = Button(newWindow, text="5. Preprocess trojan data", command=make_header_trojan_data)
    preprocessing_button5.pack()
    labelExample.pack()


def total_dataset_and_split_data():
    # global button6
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Total dataset")
    labelExample.pack()
    button4 = Button(newWindow, text="New Window button", command=file_load_event)
    button4.pack()
    button5 = Button(newWindow, text="New Window button", command=file_load_event)
    button5.pack()
    button6 = Button(newWindow, text="New Window button", command=file_load_event)
    button6.pack()

def train_save_model():
    # global button6
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Train data and save model")
    labelExample.pack()
    button4 = Button(newWindow, text="New Window button", command=file_load_event)
    button4.pack()
    button5 = Button(newWindow, text="New Window button", command=file_load_event)
    button5.pack()
    button6 = Button(newWindow, text="New Window button", command=file_load_event)
    button6.pack()

button = Button(tk, text="Preprocessing", command=preprocessing, width=10, height=5)
button2 = Button(tk, text="Total dataset", command= total_dataset_and_split_data, width=10, height=5)
button.pack(side=LEFT,padx=20,pady=20)
button2.pack(side=LEFT,padx=20,pady=40)
button3 = Button(tk, text="Train data and save model", width=15, height=5, command=train_save_model)
button3.pack(side=LEFT, padx=30, pady=40)

tk.mainloop()