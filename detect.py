import cv2
import time
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from camera import VideoStream

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

# Init Object Detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Init MiDas model
midas_model_path = "model-small.onnx"
ort_session = ort.InferenceSession(midas_model_path)
def transform_image(image):
    # Resize the image to 256x256 (required input size for MiDaS)
    image = cv2.resize(image, (256, 256))
    # Normalize the image to the range [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    # Convert HWC to CHW format (required by the model)
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension (N, C, H, W)
    image = np.expand_dims(image, axis=0)
    return image

# Init Cam
stream = VideoStream().start()

# Limit frame rate
target_fps = 10  # Set target FPS for Raspberry Pi
frame_interval = 1.0 / target_fps
frame_count = 0
frame_width = 320
frame_height = 320

thres = 0.6

while True:
    start_time = time.time()

    # Capture image
    frame = stream.read()

    # Detect object
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    
## Measure distance using MiDaS ##

    input_image = transform_image(frame)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)
    depth_map = ort_outs[0].squeeze()

    # Resize depth map to match frame size
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    # Resize depth map to match frame size
    depth_map = cv2.resize(depth_map, (frame_width, frame_height))

    # # Normalize depth map for visualization
    # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # # Blend the depth map with the original frame
    # blended_frame = cv2.addWeighted(frame, 0.6, depth_map_colored, 0.4, 0)

## Measure Distance end ##

    # Output
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = f'{labels[classId - 1]}: {confidence:.2f}'
            print(f"Detected {label}")
            # cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate the average depth value within the bounding box
            x, y, w, h = box
            depth_values = depth_map[y:y+h, x:x+w]
            avg_depth = np.mean(depth_values)
            print(f"Object: {labels[classId - 1]}, Confidence: {confidence:.2f}, Avg Depth: {avg_depth:.2f}")


    # cv2.imshow('frame', frame)
    
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    print(f"FPS: {fps:.2f}")

    # Limit frame rate (Biar Optimized, and CPU no overheat yaya)
    time_to_wait = max(0, frame_interval - elapsed_time)
    time.sleep(time_to_wait)
    
    # if cv2.waitKey(1) == ord('q'):
    #     break

# cap.release()
# cv2.destroyAllWindows()
