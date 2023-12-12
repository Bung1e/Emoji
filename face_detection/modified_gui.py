import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import threading


def exit_application():
    root.destroy()

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('D:/VS Projects/Emojis/model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
emoji_dist={0:"D:/VS Projects/Emojis/face_detection/emojis/angry1.png",2:"D:/VS Projects/Emojis/face_detection/emojis/disgusted.png",2:"D:/VS Projects/Emojis/face_detection/emojis/fearful1.png",3:"D:/VS Projects/Emojis/face_detection/emojis/happy1.png",4:"D:/VS Projects/Emojis/face_detection/emojis/neutral1.png",5:"D:/VS Projects/Emojis/face_detection/emojis/sad1.png",6:"D:/VS Projects/Emojis/face_detection/emojis/surpriced1.png"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]

def show_vid():      
    cap1 = cv2.VideoCapture(0)                                 
    if not cap1.isOpened():                             
        print("Can't open the camera1")
        return 

    global last_frame1
    while True:
        flag1, frame1 = cap1.read()
        frame1 = cv2.resize(frame1, (600, 500))

        bounding_box = cv2.CascadeClassifier(r'C:/Users/Alex/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex

        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap1.release()
            cv2.destroyAllWindows()
            break  

def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(pic2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    frame_number = 0
    root=tk.Tk()

    style = ttk.Style()
    style.configure('TButton', font=('Arial', 12, 'bold'), borderwidth='4')
    style.configure('TLabel', font=('Arial', 12, 'bold'), background='black', foreground='white')
    style.map('TButton', foreground=[('active', '!disabled', 'green')], background=[('active', 'black')])
    root.option_add('*TButton*Font', 'Arial 14 bold')
    root.option_add('*TLabel*Font', 'Arial 14')
    
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side='left')
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side='right')
    lmain2.place(x=900,y=350)
            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    title_label = ttk.Label(root, text='Emotion Recognition', font=('Arial', 20, 'bold'))
    title_label.pack(side='top', pady=5)
    style.configure('Quit.TButton', foreground='red', font=('Arial', 25, 'bold'))
    exit_button = ttk.Button(root, text='Quit', style='Quit.TButton', command=exit_application)
    exit_button.pack(side='bottom')
    threading.Thread(target=show_vid).start()
    threading.Thread(target=show_vid2).start()
    root.mainloop()