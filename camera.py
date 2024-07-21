import cv2
import numpy as np
from threading import Thread, Lock

class VideoStream:
    def __init__(self, camres=(540,270), detres=(320, 320), framerate=36, lcamidx=0, rcamidx=2):
        self.detres = detres

        self.lcam = cv2.VideoCapture(lcamidx, cv2.CAP_V4L2)
        self.lcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.lcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.lcam.set(cv2.CAP_PROP_FPS, framerate)

        self.rcam = cv2.VideoCapture(rcamidx, cv2.CAP_V4L2)
        self.rcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.rcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.rcam.set(cv2.CAP_PROP_FPS, framerate)

        self.stopped = False

        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(self.lcam, True)).start()
        Thread(target=self.update, args=(self.rcam, False)).start()
        return self

    def update(self, cam, isleft):
        while True:
            if self.stopped:
                cam.release()
                return

            ret, read = cam.read()
            if not ret:
                camstr = "left" if isleft else "right"
                print(f"[ERROR] Failed to read from {camstr} camera")
                return
            frame = cv2.resize(read, (self.detres[0], self.detres[1]/2), interpolation=cv2.INTER_NEAREST)

            with self.lock:
                if isleft:
                    self.lframe = frame
                else:
                    self.rframe = frame

    def read(self):
        with self.lock:
            return np.vstack((self.lframe, self.rframe))

    def stop(self):
        self.stopped = True
