import cv2
import numpy as np
from threading import Thread, Lock

class VideoStream:
    def __init__(self, camres=(540,270), detres=(320, 320), framerate=36, lcamidx=0, rcamidx=1):
        self.detres = detres

        self.lcam = cv2.VideoCapture(lcamidx)
        self.lcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.lcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.lcam.set(cv2.CAP_PROP_FPS, framerate)

        self.rcam = cv2.VideoCapture(rcamidx)
        self.rcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.rcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.rcam.set(cv2.CAP_PROP_FPS, framerate)

        self.lframe = None
        self.rframe = None

        self.stopped = False

        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(self.lcam, True)).start()
        Thread(target=self.update, args=(self.rcam, False)).start()
        return self

    def getframe(self, cam, isleft):
        ret, read = cam.read()
        if not ret:
            camstr = "left" if isleft else "right"
            print(f"[ERROR] Failed to read from {camstr} camera")
            exit()
        return cv2.resize(read, (self.detres[0], int(self.detres[1]/2)), interpolation=cv2.INTER_NEAREST)

    def update(self, cam, isleft):
        while True:
            if self.stopped:
                cam.release()
                return
            
            with self.lock:
                if isleft:
                    self.lframe = self.getframe(self.lcam, True)
                else:
                    self.rframe = self.getframe(self.rcam, False)

    def read(self):
        with self.lock:
            if self.lframe is None:
                self.lframe = self.getframe(self.lcam, True)
            if self.rframe is None:
                self.rframe = self.getframe(self.rcam, False)
            stacked = np.vstack((self.lframe, self.rframe))
            cv2.imwrite("outputs/ast.jpg", stacked)
            return stacked

    def stop(self):
        self.stopped = True
