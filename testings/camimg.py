import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Minimal resolution for wide aspect ratio
# Nemesis around 270x135
# DAlliance around 540x270
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

while True:
    ret, image = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")
    a = cv2.imwrite("../outputs/noriz-no.jpg", image)
    if not a:
        raise RuntimeError("failed to write frame")
