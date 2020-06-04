import numpy as np
import imutils
import cv2
import time
import dlib

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def keyPoints_trans(keyPoints):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (keyPoints.part(i).x, keyPoints.part(i).y)
    return coords

def getKeypoints(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectBB = DETECTOR(grayImage, 1)

    for (i, rect) in enumerate(rectBB):
        keyPoints = PREDICTOR(grayImage, rect)
        keyPoints = keyPoints_trans(keyPoints)
    
        return keyPoints

def findKeypoints(image):
    image = imutils.resize(image, width=500)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectBB = DETECTOR(grayImage, 1)

    for (i, rect) in enumerate(rectBB):
        (x1, y1, x2, y2) = (rect.left(), rect.top(), rect.right(), rect.bottom())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        keyPoints = PREDICTOR(grayImage, rect)
        keyPoints = keyPoints_trans(keyPoints)

        for (x, y) in keyPoints:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    return image

def image():
    img = cv2.imread('face.jpg')
    img = findKeypoints(img)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webCam():
    video = cv2.VideoCapture('/dev/video2')
    ret, frame = video.read()

    while ret:
        frame = findKeypoints(frame)
        cv2.imshow('Ouput', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        ret, frame = video.read()

if __name__ == "__main__":
    webCam()