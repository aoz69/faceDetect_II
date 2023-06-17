# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import tkinter as tk
from tkinter import messagebox
from predict import imagePredict
from model import training_model
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def message():
    # Clear previous texts written in the box
    output_text.delete(1.0, tk.END)  
    # prints text saying please wait
    output_text.insert(tk.END, "Please wait...")
    # this is to spawn alert box to give user a message
    messagebox.showinfo("Alert", "Please wait till the training is done, we will inform you shortly. Do not close other windows. Press OK to start")
    # calls function that is created below
    runFile()

# function to run necessary files and output the predictions and alert messages
def runFile():
    # calls imported function of model
    training_model()
    # Display an alert when the first file has completed running
    messagebox.showinfo("Alert", "The first file has completed running!, Press ok to start the prediction")
    pred()

def pred():
    # calls imported function for prediction
    predicted_class, confidence = imagePredict()
    # converts the console log object into stiring
    classs = str(predicted_class)
    # for searching the number of class which is after "[" and before "]"
    start_index = classs.find("[") + 1
    end_index = classs.find("]")
    # conver that string to int again
    array_value = int(classs[start_index:end_index])
    # Display the console output of the second file inside the text box
    # Clear previous texts written in the box
    output_text.delete(1.0, tk.END)  
    # gives user idea of what's going on
    output_text.insert(tk.END, "Now detecting face....")
    # Clear previous texts written in the box
    output_text.delete(1.0, tk.END) 
    # prints confidance and classs of the predicted face in the app
    output_text.insert(tk.END, "Confidance is: "+ str(confidence) + " person belongs to class: " + str(array_value))

# Create the window
window = tk.Tk()
#  name the window face detector
window.title("Face Detector")

# Create the button with message and action to call a function
button = tk.Button(window, text="Start Detection", command=message)
button.pack(pady=5)
button2 = tk.Button(window, text="Start Prediction", command=pred)
# button styling
button2.pack(pady=5)

# Create the text box and give some styles to it
output_text = tk.Text(window, height=20, width=70, bg="black", fg="green")
output_text.pack()

# Start the Tkinter event loop
window.mainloop()