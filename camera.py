import cv2
import numpy as np
from threading import Thread, Lock
import logging

class VideoStream:
    def __init__(self, camres=(540, 270), detres=(320, 320), framerate=36, lcamidx=0, rcamidx=2):
        self.detres = detres
        self.stopped = False
        self.lock = Lock()

        # Initialize left camera
        self.lcam = cv2.VideoCapture(lcamidx, cv2.CAP_V4L2)
        if not self.lcam.isOpened():
            raise ValueError(f"Failed to open left camera (index {lcamidx})")

        self.lcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.lcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.lcam.set(cv2.CAP_PROP_FPS, framerate)

        # Initialize right camera
        self.rcam = cv2.VideoCapture(rcamidx, cv2.CAP_V4L2)
        if not self.rcam.isOpened():
            raise ValueError(f"Failed to open right camera (index {rcamidx})")

        self.rcam.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
        self.rcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
        self.rcam.set(cv2.CAP_PROP_FPS, framerate)

        # Capture initial frames
        self.lframe = self._capture_initial_frame(self.lcam, "left")
        self.rframe = self._capture_initial_frame(self.rcam, "right")

    def _capture_initial_frame(self, cam, camstr):
        ret, read = cam.read()
        if not ret:
            logging.error(f"No frame received from {camstr} camera")
            raise RuntimeError(f"Failed to capture initial frame from {camstr} camera")

        # Resize if necessary
        frame_height, frame_width = read.shape[:2]
        if (frame_width, frame_height) != self.detres:
            read = cv2.resize(read, (self.detres[0], int(self.detres[1] / 2)), interpolation=cv2.INTER_NEAREST)
        return read

    def start(self):
        self.stopped = False
        self.lthread = Thread(target=self.update, args=(self.lcam, True))
        self.rthread = Thread(target=self.update, args=(self.rcam, False))
        self.lthread.start()
        self.rthread.start()
        return self

    def update(self, cam, isleft):
        while not self.stopped:
            ret, read = cam.read()
            if not ret:
                camstr = "left" if isleft else "right"
                logging.error(f"Failed to read from {camstr} camera")
                self.stop()
                return

            # Resize outside the critical section if necessary
            frame_height, frame_width = read.shape[:2]
            if (frame_width, frame_height) != self.detres:
                read = cv2.resize(read, (self.detres[0], int(self.detres[1] / 2)), interpolation=cv2.INTER_NEAREST)

            # Lock only when updating the shared resource
            with self.lock:
                if isleft:
                    self.lframe = read
                else:
                    self.rframe = read

    def read(self):
        with self.lock:
            return np.vstack((self.lframe, self.rframe))

    def stop(self):
        self.stopped = True
        self.lthread.join()
        self.rthread.join()
        self.lcam.release()
        self.rcam.release()
        logging.info("Cameras released and threads stopped.")