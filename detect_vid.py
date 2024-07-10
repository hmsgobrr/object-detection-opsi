import torchvision
import cv2
import torch
import argparse
import time
import detect_utils

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
args = vars(parser.parse_args())

if torch.cuda.is_available():
    print("ada kudaaaaaaaa")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model 
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

model = model.eval().to(device)

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}"

out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.7)
        
        # draw boxes and show current frame on screen
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)

        # end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # fps
        frame_count += 1
        # write the FPS on frame
        cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # press `q` to exit
        wait_time = max(1, int(fps/4))
        # chane color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
