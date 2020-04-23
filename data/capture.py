import imutils
import os
import uuid
from cv2 import cv2

cap = cv2.VideoCapture(0)
cheatsheet = cv2.imread('alfabetet.png')

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for letter in alphabet:
    count = 0
    countDown = 10
    while(True):
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        top, right, bottom, left = 170, 200, 385, 440
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 1)
        text = 'Letter: ' + letter.upper() + '(' + str(countDown) + ')'
        cv2.putText(frame, text, (right, top-5), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1)
        cv2.imshow('frame', frame)
        cv2.imshow('Cheatsheet', cheatsheet)
        path = os.path.join(os.getcwd(), letter)
        if (os.path.exists(path) == False):
            os.mkdir(path)
        k = cv2.waitKey(33)
        if k==32:
            filename = os.path.join(path, str(uuid.uuid4().hex)) + '.jpg'
            print('creating file: ' + filename)
            roi = frame[top:bottom, right:left]
            print('Saving file: ', filename)
            cv2.imwrite(filename, roi)
            count = count + 1
            countDown = countDown - 1
        if count == 10:
            break
cap.release()
cv2.destroyAllWindows()
