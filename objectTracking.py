from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument('-v', '--video', type=str)
# args = vars(ap.parse_args())

tracker = cv2.TrackerMOSSE_create()
initBB = None

print("Starting Video Stream.....")
vs = cv2.VideoCapture(0)
time.sleep(1)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        initBB = cv2.selectROI("Frame", frame)
        tracker.init(frame, initBB)

    if key == ord('q'):
        vs.release()
        cv2.destroyAllWindows()
        break

    if initBB is not None:
        (success, bbox) = tracker.update(frame)
        print(bbox)

        if success:
            (x, y, w, h) = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)