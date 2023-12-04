from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os


tk = Tk()
tk.title("Trojan!!!")
tk.geometry("600x400+100+100")
tk.resizable(True,True)
label = Label(tk, text="Trojan program")

label.pack(side=TOP, padx=10, pady=10)
number = 0
file_path = ""
folder_name = ""
preprocessing_button1 = Button()
preprocessing_button2 = Button()
preprocessing_button3 = Button()

def event():
    global number
    number += 1
    button['text'] = '버튼 누른횟수: ' + str(number)

    
def file_load_event():
    global file_path
    file_path = filedialog.askdirectory()
    print(file_path)
    name_list = os.listdir(file_path)
    path_name = []
    for i in range(len(name_list)):
        path_name.append(file_path + name_list[i])
    print(path_name)
    
def make_preprocessing_folder():
    global preprocessing_button1
    global folder_name
    folder_name = "/preprocessing"
    global file_path
    file_path = filedialog.askdirectory()
    folder_name = file_path + folder_name
    # folder 없어도 넘어감
    os.makedirs(folder_name, exist_ok=True)
    messagebox.showinfo("Message", "Success for making preprocessing folder!!")
    preprocessing_button1['text'] = "Success for making folder"
def preprocess_trusted_data():
    messagebox.showinfo("Message", "Success for preprocessing trusted data")
    preprocessing_button1['text'] = "Success for preprocessing trusted data"

def preprocess_trojan_data():
    messagebox.showinfo("Message", "Success for preprocessing trojan data")
    preprocessing_button1['text'] = "Success for preprocessing trojan data"

def preprocessing():
    global preprocessing_button1, preprocessing_button2, preprocessing_button3
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Preprocessing")
    preprocessing_button1 = Button(newWindow, text="Make preprocessing folder", command=make_preprocessing_folder)
    preprocessing_button1.pack()
    preprocessing_button2 = Button(newWindow, text="Preprocess trusted data", command=preprocess_trusted_data)
    preprocessing_button2.pack()
    preprocessing_button3 = Button(newWindow, text="Preprocess trojan data", command=preprocess_trojan_data)
    preprocessing_button3.pack()
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