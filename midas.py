import cv2
import time
import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

# Load the object detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Load MiDas model
midas_model_path = "model-small.onnx"
ort_session = ort.InferenceSession(midas_model_path)

# Preprocessing function for MiDas
def transform_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = transform(image).unsqueeze(0)
    return image.numpy()

# Open video capture
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

started = time.time()
last_logged = time.time()
frame_count = 0

thres = 0.56

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    
    # Perform depth estimation
    input_image = transform_image(frame)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)
    depth_map = ort_outs[0].squeeze()

    # Resize depth map to match frame size
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend the depth map with the original frame
    blended_frame = cv2.addWeighted(frame, 0.6, depth_map_colored, 0.4, 0)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = f'{labels[classId - 1]}: {confidence:.2f}'
            cv2.rectangle(blended_frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(blended_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate the average depth value within the bounding box
            x, y, w, h = box
            depth_values = depth_map[y:y+h, x:x+w]
            avg_depth = np.mean(depth_values)
            print(f"Object: {labels[classId - 1]}, Confidence: {confidence:.2f}, Avg Depth: {avg_depth:.2f}")

    # Display the result
    cv2.imwrite('outputs/midascap.jpg', blended_frame)
    
    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        fps = frame_count / (now - last_logged)
        print(f"{fps:.2f} fps")
        last_logged = now
        frame_count = 0
    
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()