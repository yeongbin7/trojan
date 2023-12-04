from tkinter import *
from tkinter import filedialog
tk = Tk()
tk.title("Trojan!!!")
tk.geometry("600x400+100+100")
tk.resizable(True,True)
label = Label(tk, text="Trojan program")

label.pack(side=TOP, padx=10, pady=10)
number = 0
file_path = ""
def event():
    global number
    number += 1
    button['text'] = '버튼 누른횟수: ' + str(number)

def file_load_event():
    global file_path
    file_path = filedialog.askdirectory()
    print(file_path)
    return file_path

def preprocessing():
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")

    labelExample = Label(newWindow, text="Preprocessing")
    button1 = Button(newWindow, text="New Window button", command=file_load_event)
    button1.pack()
    button2 = Button(newWindow, text="New Window button", command=file_load_event)
    button2.pack()
    button3 = Button(newWindow, text="New Window button", command=file_load_event)
    button3.pack()
    labelExample.pack()


def total_dataset_and_split_data():
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Total dataset")
    labelExample.pack()
    button1 = Button(newWindow, text="New Window button", command=file_load_event)
    button1.pack()
    button2 = Button(newWindow, text="New Window button", command=file_load_event)
    button2.pack()
    button3 = Button(newWindow, text="New Window button", command=file_load_event)
    button3.pack()

def train_save_model():
    newWindow = Toplevel(tk)
    newWindow.geometry("200x200+100+100")
    labelExample = Label(newWindow, text="Train data and save model")
    labelExample.pack()
    button1 = Button(newWindow, text="New Window button", command=file_load_event)
    button1.pack()
    button2 = Button(newWindow, text="New Window button", command=file_load_event)
    button2.pack()
    button3 = Button(newWindow, text="New Window button", command=file_load_event)
    button3.pack()

button = Button(tk, text="Preprocessing", command=preprocessing, width=10, height=5)
button2 = Button(tk, text="Total dataset", command= total_dataset_and_split_data, width=10, height=5)
button.pack(side=LEFT,padx=20,pady=20)
button2.pack(side=LEFT,padx=20,pady=40)
button3 = Button(tk, text="Train data and save model", width=15, height=5, command=train_save_model)
button3.pack(side=LEFT, padx=30, pady=40)


tk.mainloop()