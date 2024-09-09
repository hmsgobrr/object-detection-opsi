import cv2
import pyttsx3
import time
import argparse
import numpy as np
from threading import Thread
from statistics import mean

# Argument parsing
parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--novoice', action='store_true')
parser.add_argument('-v', '--video', type=str, required=True, help="Path to the input video file")
args = parser.parse_args()

# NO_VOICE = args.novoice
VIDEO_PATH = args.video

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
engine.setProperty('rate', 150)


# Load the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print(f"Error: Unable to open video file {VIDEO_PATH}")
    exit()

fpses = []
run = True

def listenexit():
    global run
    while run:
        i = input()
        if i == "q":
            run = False
            print("## AVERAGE FPS:", mean(fpses))

exthd = Thread(target=listenexit)
exthd.start()

save_name = f"{VIDEO_PATH.split('/')[-1].split('.')[0]}"

frame_height, frame_width = 1280, 720

out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (320, 320))

started = time.time()
last_logged = time.time()
frame_count = 0
fpses = []
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# Video processing loop
while run and video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # End of video

    startdet = time.time()
    
    # Split the fraem
    skibiframe = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_NEAREST)
    top_frame = skibiframe[:, :320 // 2]
    bottom_frame = skibiframe[:, 320 // 2:]
    combined_frame = np.vstack((top_frame, bottom_frame))

    
    # Perform object detection on both top and bottom halfse
    # detecsreng = {
    #     'in front of you': {'distant': {}, 'close': {}},
    #     'on your right': {'distant': {}, 'close': {}},
    #     'on your left': {'distant': {}, 'close': {}}
    # }
    classIds, confs, bbox = net.detect(combined_frame, confThreshold=0.56)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box
            cx = x + w / 2
            cy = y + h / 2

            # dist = -1
            # if labels[classId - 1] in KNOWN_WIDTHS.keys():
            #     focal_len = FOCLEN_DA if y > frame_height / 2 else FOCLEN_NS
            #     dist = (KNOWN_WIDTHS[labels[classId - 1]] * focal_len) / w

            label = f'{labels[classId - 1]}: {confidence:.2f}'
            print(f"Detected {label}")
            # if dist != -1:
            #     print(f"\tAt distance: {dist} meters")
            
            # # Update the detection registry based on object position
            # position = 'in front of you'  # Default to center
            # if cx > frame_width / 2:
            #     position = 'on your right'
            # elif cx < frame_width / 2:
            #     position = 'on your left'

            # detecsreng[position]['distant' if dist > 1000 else 'close'][labels[classId - 1]] = \
            #     detecsreng[position]['distant' if dist > 1000 else 'close'].get(labels[classId - 1], 0) + 1

            # Draw rectangle and label
            cv2.rectangle(combined_frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(combined_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print("\t## DETECT P. DELAY", time.time() - startdet)

    # if not NO_VOICE:
    #     for direction in detecsreng.keys():
    #         for range_category in detecsreng[direction]:
    #             if len(detecsreng[direction][range_category]) < 1:
    #                 continue
    #             speech = ""
    #             for obj in detecsreng[direction][range_category].keys():
    #                 speech += f"{detecsreng[direction][range_category][obj]} {obj}, "
    #             speech += f"{range_category} {direction}"
    #             engine.say(speech)
    #             engine.runAndWait()

    out.write(combined_frame)

    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        fps = frame_count / (now - last_logged)
        print(f"{fps:.2f} fps")
        fpses.append(fps)
        last_logged = now
        frame_count = 0

    # if cv2.waitKey(1) == ord('q'):
    #     break

# Release resources
print("## AVG FPS: ", sum(fpses)/len(fpses))
video.release()
# cv2.destroyAllWindows()