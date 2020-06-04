from Facial_Landmarks import getKeypoints
from scipy.spatial import distance
import numpy as np
import imutils
import cv2
import time
import os

duration = 5
freq = 440
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0
TOTAL = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A+B)/(2.0*C)

def webCam():
    video = cv2.VideoCapture('/dev/video2')
    ret, frame = video.read()

    global COUNTER, TOTAL
    
    while ret:
        frame = imutils.resize(frame, width=500)
        keyPoints = getKeypoints(frame)

        if keyPoints is not None:
            left_eye, right_eye = keyPoints[36:42], keyPoints[42:48]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye))/2.0

            left_eye_Hull, right_eye_Hull = cv2.convexHull(left_eye), cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_Hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_Hull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                if COUNTER == EYE_AR_CONSEC_FRAMES:
                    COUNTER = 0
                    TOTAL += 1
                else:
                    COUNTER += 1

            else:
                COUNTER = max(0, COUNTER-3)

            cv2.putText(frame, "EAR : {}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Ouput', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        ret, frame = video.read()

if __name__ == "__main__":
    webCam()