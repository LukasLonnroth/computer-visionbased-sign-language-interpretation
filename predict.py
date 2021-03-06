import imutils
import numpy as np
import os
import tensorflow as tf
from cv2 import cv2

cap = cv2.VideoCapture(0)
cheatsheet = cv2.imread('alfabetet.png')

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
predicted_letter = ''
text = 'Please make a sign and press space'
model_name = 'pretrained_models/grayscale_64-64_128drop05_256_512_dense_64_1586770972.model'
model = tf.keras.models.load_model(model_name)
spelled_letters = ''
last_letter = ''
print('Tensorflow version: {}'.format(tf.__version__))

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    top, right, bottom, left = 170, 200, 385, 440
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.putText(frame, text, (right, top-5), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1)
    cv2.putText(frame, spelled_letters, (right, top-20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1)
    cv2.imshow(model_name, frame)
    k = cv2.waitKey(33)
    roi = frame[top:bottom, right:left]
    roi = cv2.resize(roi, (64, 64))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow('ROI', imutils.resize(gray, width=128))
    array = np.array(gray).reshape(-1, 64, 64, 1)
    normalized = array / 255.0
    confidence = np.amax(model.predict(normalized))
    prediction = model.predict_classes(normalized)
    letter = alphabet[prediction[0]].upper()
    text = 'Prediction: ' + letter + ' Confidence: ' + str(round(confidence, 2))
    if k==32:
        spelled_letters = spelled_letters + alphabet[prediction[0]]
        last_letter = letter
    if k==99:
        spelled_letters = ''
        last_letter = ''
    if k==113:
        break

cap.release()
cv2.destroyAllWindows()

