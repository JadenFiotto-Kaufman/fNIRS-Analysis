from src.main.fNIR import fNIR
import os
from tkinter import ttk
from tkinter.filedialog import askopenfilename
if __name__ == "__main__":
    fNIR.train("../../processed/", "DNN", epochs=2000, batch_size=3, combine=True, test_size=.1, load=False, scale=True)
    #name = askopenfilename(filetypes=(("Text File", "*.txt"), ("All Files", "*.*")),title="Choose a file.")
    #fNIR.predict(name)