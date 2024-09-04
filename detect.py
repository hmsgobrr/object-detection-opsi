import cv2
import pyttsx3
import time
from camera import VideoStream

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

KNOWN_WIDTHS = {
    'couch': 145,
    'person': 50,
    'backpack': 40,
    'chair': 60,
    'bottle': 8,
    'car': 300
}
FOCLEN_DA = 170
FOCLEN_NS = 215

engine = pyttsx3.init(driverName='espeak')
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[39])
engine.setProperty('rate', 150)

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
    frame = stream.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    detecsreng = {'distant': {}, 'close': {}}
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box

            dist = -1
            if labels[classId - 1] in KNOWN_WIDTHS.keys():
                focal_len = FOCLEN_DA if y > 160 else FOCLEN_NS
                dist = (KNOWN_WIDTHS[labels[classId - 1]]*focal_len)/w

            label = f'{labels[classId - 1]}: {confidence:.2f}'
            print(f"Detected {label}")
            if dist != -1:
                print(f"\tAt distance: {dist} meters")
            
            detecsreng['far' if dist > 1000 else 'close'][labels[classId - 1]] = detecsreng['far' if dist > 1000 else 'close'].get(labels[classId - 1],0)+1
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(f"outputs/{labels[classId - 1]}.jpg", frame)
    for distens in detecsreng.keys():
        speech = ""
        for obj in detecsreng[distens].keys():
            speech += f"{detecsreng[distens][obj]} {obj}, "
        speech += " " + distens
        engine.runAndWait(speech)

    cv2.imwrite("outputs/detectboxout.jpg", frame)

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
