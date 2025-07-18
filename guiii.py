import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import model_from_json

# Emotion labels (ensure these match your model)
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r", encoding='utf-8') as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    return model

# Load the model
model = FacialExpressionModel("model.json", "model_weights.weights.h5")
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# GUI Setup
top = tk.Tk()
top.title("Emotion Detector")
top.geometry("800x600")
label1 = Label(top, background="#CDCDCD", font=("Helvetica", 20))
label2 = Label(top, background="#CDCDCD", font=("Helvetica", 20))

webcam_label = Label(top)
webcam_label.pack()

label1.pack()
label2.pack()

# Video processing function
cap = cv2.VideoCapture(0)

def show_frames():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        maxindex = int(np.argmax(prediction))
        predicted_emotion = EMOTIONS_LIST[maxindex]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame in GUI
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)
    webcam_label.after(10, show_frames)

show_frames()
top.mainloop()
