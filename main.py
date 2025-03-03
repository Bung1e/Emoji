import cv2
from model import FacialExpressionModel
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

facec = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Emojis/haarcascade_frontalface_default.xml'))
model = FacialExpressionModel(
    os.path.join(BASE_DIR, 'Emojis/model.json'),
    os.path.join(BASE_DIR, 'Emojis/model_weights.h5')
)

font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return fr

def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

gen(VideoCamera())
