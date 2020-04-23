from cv2 import cv2
import os
import glob
import uuid

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for letter in alphabet:
    current_dir = os.getcwd()
    path = os.path.join(current_dir, letter)

    for filename in glob.glob(path + "/*.jpg"):
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (128,128))
        train_path = os.path.join(current_dir, 'handalfabet_korpus_9_4_2020')
        if (os.path.exists(train_path) == False):
            os.mkdir(train_path)
        letter_path = os.path.join(train_path, letter)
        if (os.path.exists(letter_path) == False):
            os.mkdir(letter_path)
        file = str(uuid.uuid4().hex) + '.jpg'
        cv2.imwrite(os.path.join(letter_path, file), img)

