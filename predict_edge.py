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
model = tf.keras.models.load_model('/Users/lukas/Documents/slutarbete2/kod/models/canny_64-64_128drop05_256_512_dense_64_1586770532.model')
spelled_letters = ''
last_letter = ''

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    top, right, bottom, left = 170, 200, 385, 440
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.putText(frame, text, (right, top-5), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1)
    cv2.putText(frame, spelled_letters, (right, top-20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(33)
    roi = frame[top:bottom, right:left]
    roi = cv2.resize(roi, (64, 64))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 90)
    cv2.imshow('ROI', imutils.resize(edge, width=200))
    #cv2.moveWindow('ROI', 20,20)
    array = np.array(edge).reshape(-1, 64, 64, 1)
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

