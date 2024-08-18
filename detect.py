import cv2
import time
from camera import VideoStream

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

# Load the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

stream = VideoStream().start()

started = time.time()
last_logged = time.time()
frame_count = 0

thres = 0.56

while True:
    classIds, confs, bbox = net.detect(stream.read(), confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            boxw = int(box[2])-int(box[0])
            boxh = int(box[3])-int(box[1])
            label = f'{labels[classId - 1]}: {confidence:.2f}, CVD {boxw*boxh*100/(51200)}%'
            print(f"Detected {label}")
            # cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.imshow('frame', frame)
    
    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        fps = frame_count / (now - last_logged)
        print(f"{fps:.2f} fps")
        last_logged = now
        frame_count = 0
    
    # if cv2.waitKey(1) == ord('q'):
    #     break

# cap.release()
# cv2.destroyAllWindows()
