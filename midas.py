import cv2
import time
import numpy as np
import onnxruntime as ort

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

# Initialize Object Detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize MiDaS model
midas_model_path = "model-small.onnx"
ort_session = ort.InferenceSession(midas_model_path)

# Image transformation function using OpenCV and NumPy
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

# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Optimize frame size and performance settings
frame_width = 320
frame_height = 240
thres = 0.60  # Increased confidence threshold to reduce false positives
frame_count = 0

# Limit frame rate
target_fps = 10  # Set target FPS for Raspberry Pi
frame_interval = 1.0 / target_fps

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Detect objects
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    ## Measure distance using MiDaS ##
    input_image = transform_image(frame)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)
    depth_map = ort_outs[0].squeeze()

    # Resize depth map to match frame size
    depth_map = cv2.resize(depth_map, (frame_width, frame_height))

    # Visualize detection results
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box
            depth_values = depth_map[y:y+h, x:x+w]
            avg_depth = np.mean(depth_values)
            label = f'{labels[classId - 1]}: {confidence:.2f}, Depth: {avg_depth:.2f}m'

            # Draw bounding box and label on the frame
            # cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Object: {labels[classId - 1]}, Confidence: {confidence:.2f}, Avg Depth: {avg_depth:.2f}")

    # Display the frame with detections
    # cv2.imshow('THE', frame)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    print(f"FPS: {fps:.2f}")

    # Limit frame rate (Biar Optimized, and CPU no overheat yaya)
    time_to_wait = max(0, frame_interval - elapsed_time)
    time.sleep(time_to_wait)

    # Exit on pressing 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
# cap.release()
# cv2.destroyAllWindows()