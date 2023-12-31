from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import glob
import os
from sklearn.decomposition import PCA
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# main screen
tk = Tk()
tk.title("Trojan!!!")
tk.geometry("450x200+100+100")
tk.resizable(True,True)
label = Label(tk, text="Trojan program")
label.pack(side=TOP)

number = 0
processing_file_path = ""
preprocessing_folder_path = ""
preprocessing_button1 = Button()
preprocessing_button2 = Button()
preprocessing_button3 = Button()
preprocessing_button4 = Button()
preprocessing_button5 = Button()
min_num = 0
max_num = 0
train_folder_path = ""
test_folder_path = ""
pre_trust_folder_path = ""
pre_trojan_file_path = ""
train_folder_path = ""
test_folder_path = ""

model = Sequential()
split_entry = Entry()
# def file_load_event():
#     global processing_file_path
#     processing_file_path = filedialog.askdirectory()
#     print(processing_file_path)
#     name_list = os.listdir(processing_file_path)
#     path_name = []
#     for i in range(len(name_list)):
#         path_name.append(processing_file_path + name_list[i])
#     print(path_name)


# making_preprocessing_folder: a function to save preprocessing data to a folder
def make_preprocessing_folder():
    global preprocessing_button1
    global preprocessing_folder_path
    processing_dir = filedialog.askdirectory()
    preprocessing_folder_path = processing_dir +  "/preprocessing"
    # exist_ok=True: If you already have the folder, pass the code line
    os.makedirs(preprocessing_folder_path, exist_ok=True)
    os.makedirs(preprocessing_folder_path + "/trojan", exist_ok=True)
    os.makedirs(preprocessing_folder_path + "/trust_data", exist_ok=True)
    
    messagebox.showinfo("Message", "Success for making preprocessing folder!!")
    preprocessing_button1['text'] = "Success for making folder"

# make_header_trusted_data: a function to make header file for many csv files of the trusted data
def make_header_trusted_data():
    trust_dir = filedialog.askdirectory() + "/"
    # Only trusted folder name not total directory
    trust_name_list = os.listdir(trust_dir)
    path_trust_name = []
    # Make total file name including directory
    for i in range(len(trust_name_list)):
        path_trust_name.append(trust_dir + trust_name_list[i])
    for i in range(len(path_trust_name)):
        sub_list = []
        file_free = glob.glob(os.path.join(path_trust_name[i], "*.csv"))
        for f in file_free:
            sub_list.append(pd.read_csv(f))
        data_free_raw = pd.concat(sub_list[:], axis=1)
        data_free_amp = (data_free_raw.drop(columns='Time', axis=1)).T
        output_path = path_trust_name[i] + '.h5'
        data_free_amp.to_hdf(output_path, 'a')

    messagebox.showinfo("Message", "Success for making header file from trusted data")
    preprocessing_button2['text'] = "Success for making header file from trusted data"

# make_header_trojan_data: a function to make header file for many csv files of the trojan data
def make_header_trojan_data():
    trojan_dir = filedialog.askdirectory() + "/"
    trojan_name_list = os.listdir(trojan_dir)
    path_trojan_name = []
    for i in tqdm(range(len(trojan_name_list))):
        path_trojan_name.append(trojan_dir + trojan_name_list[i])
    for i in tqdm(range(len(path_trojan_name))):
        sub_list = []
        file_free = glob.glob(os.path.join(path_trojan_name[i], "*.csv"))
        for f in file_free:
            sub_list.append(pd.read_csv(f))
        data_free_raw = pd.concat(sub_list[:], axis=1)
        data_free_amp = (data_free_raw.drop(columns='Time', axis=1)).T
        output_path = path_trojan_name[i] + '.h5'
        data_free_amp.to_hdf(output_path, 'a')

    messagebox.showinfo("Message", "Success for making header file from trojan data")
    preprocessing_button3['text'] = "Success for making header file from trojan data"

# Preprocess_trusted_data: a function to preprocess the data using data cutting and save the file.
def Preprocess_trusted_data():
    global min_num, max_num
    global preprocessing_folder_path
    trust_file_list = []
    trust_file_dir = glob.glob(os.path.join(filedialog.askdirectory(), "*.h5"))
    for i in tqdm(trust_file_dir):
        trust_file_list.append(pd.read_hdf(i))
    # Data cutting process
    min_num = 0
    max_num = 0
    # You can adjust the cutting value if you want to change the range
    cutting_value = 0.025
    for i in tqdm(range(len(trust_file_list))):
        trust_mean_data = trust_file_list[i].mean()
        trust_mean_data = trust_mean_data[trust_mean_data >= cutting_value]
        s = trust_mean_data.to_frame()
        if i == 0:
            min_num = s.index[0]
            max_num = s.index[-1]
        else:
            if min_num < s.index[0]:
                min_num = s.index[0]
            if max_num > s.index[-1]:
                max_num = s.index[-1]
    
    for i in tqdm(range(len(trust_file_list))):
        trust_file_list[i] = trust_file_list[i].iloc[:,min_num:max_num]
        trust_file_list[i].columns = range(1,max_num - min_num + 1)
        tru_dir = preprocessing_folder_path + "/trust_data/" + trust_file_dir[i].split('\\')[1]
        trust_file_list[i].to_hdf(tru_dir, 'a')
        print(" Save {} \n".format(trust_file_dir[i].split('\\')[1]))
    messagebox.showinfo("Message", "Success for preprocessing trusted data")
    preprocessing_button4['text'] = "Success for preprocessing trusted data"

# Preprocess_trojan_data: a function to preprocess the data using the range coming out of the trusted data and save the file.
def Preprocess_trojan_data():
    global min_num, max_num
    global preprocessing_folder_path
    trojan_file_list = []
    trojan_file_dir = glob.glob(os.path.join(filedialog.askdirectory(), "*.h5"))
    for i in trojan_file_dir:
        trojan_file_list.append(pd.read_hdf(i))
    for i in tqdm(range(len(trojan_file_list))):
        trojan_file_list[i] = trojan_file_list[i].iloc[:,min_num:max_num]
        trojan_file_list[i].columns = range(1,max_num - min_num + 1)
        troj_dir = preprocessing_folder_path + "/trojan/" + trojan_file_dir[i].split('\\')[1]
        trojan_file_list[i].to_hdf(troj_dir, 'a')
        print(" Save {} \n".format(trojan_file_dir[i].split('\\')[1]))
    messagebox.showinfo("Message", "Success for preprocessing trojan data")
    preprocessing_button5['text'] = "Success for preprocessing trojan data"

# preprocessing: a function to show the GUI for the all functions of the preprocessing.
def preprocessing():
    global preprocessing_button1, preprocessing_button2, preprocessing_button3
    newWindow = Toplevel(tk)
    newWindow.geometry("400x200+100+100")
    labelExample = Label(newWindow, text="Preprocessing")
    preprocessing_button1 = Button(newWindow, text="1. Make preprocessing folder", command=make_preprocessing_folder)
    preprocessing_button1.pack()
    preprocessing_button2 = Button(newWindow, text="2. Make header as trusted csv data", command=make_header_trusted_data)
    preprocessing_button2.pack()
    preprocessing_button3 = Button(newWindow, text="3. Make header as trojan csv data", command=make_header_trojan_data)
    preprocessing_button3.pack()
    preprocessing_button4 = Button(newWindow, text="4. Preprocess trusted data", command=Preprocess_trusted_data)
    preprocessing_button4.pack()
    preprocessing_button5 = Button(newWindow, text="5. Preprocess trojan data", command=Preprocess_trojan_data)
    preprocessing_button5.pack()
    labelExample.pack()

# train_folder_path: a function for setting train folder path
def train_folder_path():
    global train_folder_path
    train_folder_path = filedialog.askdirectory() + "/"

# test_folder_path: a function for setting test folder path
def test_folder_path():
    global test_folder_path
    test_folder_path = filedialog.askdirectory() + "/"

# preprocessing_trust_folder_path: a function for setting folder path including preprocessed trusted data
def preprocessing_trust_folder_path():
    global pre_trust_folder_path
    pre_trust_folder_path = filedialog.askdirectory()

# preprocessing_trojan_file_path: a function for setting file path including preprocessed trojan data
def preprocessing_trojan_file_path():
    global pre_trojan_file_path
    pre_trojan_file_path = filedialog.askopenfilename()

# split_data: a function to make trusted and trojan data as one data, split the data as train and test using split factor, and save the file respectively.
def split_data():
    global train_folder_path, test_folder_path, pre_trust_folder_path, pre_trojan_file_path
    total_data_list = []
    trust_data_list = os.listdir(pre_trust_folder_path)
    for i in range(len(trust_data_list)):
        total_data_list.append(pd.read_hdf(pre_trust_folder_path+"/"+ trust_data_list[i]))
    troj_df = pd.read_hdf(pre_trojan_file_path)
    total_data_list.append(troj_df)

    for i in range(len(total_data_list)):
        total_data_list[i]['label'] = i+1
    total_data = pd.concat(total_data_list, ignore_index = True)
    total_data = total_data.astype(dtype = 'float32', errors = 'ignore')
    y_data = total_data['label']
    x_data = total_data.drop(labels='label', axis=1)

    #Split the data
    split_factor = split_entry.get()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data , test_size = float(split_factor), stratify=y_data) 

    outputpath1= train_folder_path + 'x_train.h5'
    x_train.to_hdf(outputpath1,'a')

    outputpath2= test_folder_path + 'x_test.h5'
    x_test.to_hdf(outputpath2,'a')

    outputpath3= train_folder_path + 'y_train.h5'
    y_train.to_hdf(outputpath3,'a')

    outputpath4= test_folder_path + 'y_test.h5'
    y_test.to_hdf(outputpath4,'a')


# total_dataset_and_split_data: a function to show GUI for the all functions of making total dataset and spliting the data
def total_dataset_and_split_data():
    global split_entry
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Total dataset")
    labelExample.pack()
    button4 = Button(newWindow, text="select_pre_trust_folder", command=preprocessing_trust_folder_path)
    button4.pack()
    button5 = Button(newWindow, text="select_pre_trojan_file", command=preprocessing_trojan_file_path)
    button5.pack()
    button6 = Button(newWindow, text="select train folder", command=train_folder_path)
    button6.pack()
    button7 = Button(newWindow, text="select test folder", command=test_folder_path)
    button7.pack()
    split_entry = Entry(newWindow)
    split_entry.pack()
    button8 = Button(newWindow, text="split data", command=split_data)
    button8.pack()

# reset_keras: a function for intialization of the keras
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    try:
        del classifier
    except:
        pass
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))

# train_data_and_test: a function for training and testing data
def train_data_and_test():
    global train_folder_path, test_folder_path
    global model
    x_train = pd.read_hdf(train_folder_path + 'x_train.h5').values    
    x_test = pd.read_hdf(test_folder_path + 'x_test.h5').values
    y_train = pd.read_hdf(train_folder_path + 'y_train.h5').values
    y_train = to_categorical(y_train)
    y_test = pd.read_hdf(test_folder_path + 'y_test.h5').values
    y_test = to_categorical(y_test)

    model = Sequential([
        Dense(units = y_train.shape[1], input_dim = x_train.shape[1] , activation = 'softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    callback = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=10,verbose=1,restore_best_weights=True)
    reset_keras()
    # You can set the epochs

    history = model.fit(x_train, y_train, batch_size = 512, epochs = 100, callbacks = [callback] ,validation_data = (x_test, y_test))
    y_pred = model.predict(x_test)
    messagebox.showinfo("Message", "Success for training and testing data")

# save_model: a function for saving model
def save_model():
    model_path = filedialog.askdirectory() 
    model.save(model_path + "/backdoor.h5")

# train_save_model: a function to show GUI for training the data and saving the model
def train_save_model():
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Train data and save model")
    labelExample.pack()
    button4 = Button(newWindow, text="Select train folder", command=train_folder_path)
    button4.pack()
    button5 = Button(newWindow, text="Select test folder", command=test_folder_path)
    button5.pack()
    button6 = Button(newWindow, text="Train data and Test", command=train_data_and_test)
    button6.pack()
    button7 = Button(newWindow, text="Save model", command=save_model)
    button7.pack()

button = Button(tk, text="Preprocessing", command=preprocessing, height=3)
button2 = Button(tk, text="Total dataset", command= total_dataset_and_split_data, height=3)
button.pack(side=LEFT,padx=20,pady=20)
button2.pack(side=LEFT,padx=20,pady=40)
button3 = Button(tk, text="Train data and save model", height=3, command=train_save_model)
button3.pack(side=LEFT, padx=30, pady=40)

tk.mainloop()