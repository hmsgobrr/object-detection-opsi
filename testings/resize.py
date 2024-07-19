import cv2

img = cv2.imread("../outputs/noriz.jpg")
res = cv2.resize(img, (224, 112), interpolation=cv2.INTER_LINEAR)
bord = cv2.copyMakeBorder(res, 56, 56, 0, 0, cv2.BORDER_CONSTANT)
if not cv2.imwrite("../outputs/noriz-bord.jpg", bord):
    raise RuntimeError("failed to write frame")
