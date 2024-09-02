import cv2
import time
# import numpy as np
# import onnxruntime as ort

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

frame_width = 320
frame_height = 160
det_input_size = max(frame_width, frame_height)

# Initialize Object Detection model (MobileNet SSD)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(det_input_size, det_input_size)  # Reduced input size for faster processing
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Load FastDepth model
# fastdepth_model_path = "model-small.onnx"
# ort_session = ort.InferenceSession(fastdepth_model_path)

# def transform_image(image):
#     image = cv2.resize(image, (256, 256))  # Keep the input size for FastDepth
#     image = image.astype(np.float32) / 255.0
#     image = np.transpose(image, (2, 0, 1))
#     image = np.expand_dims(image, axis=0)
#     return image

# Initialize camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

thres = 0.60
target_fps = 10
frame_interval = 1.0 / target_fps
skip_frames = 2  # Skip every 2nd frame for efficiency

def process_frame(frame):
    # Object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    # Depth estimation
    # input_image = transform_image(frame)
    # ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    # ort_outs = ort_session.run(None, ort_inputs)
    # depth_map = ort_outs[0].squeeze()
    # depth_map = cv2.resize(depth_map, (frame_width, frame_height))

    # Draw bounding boxes and depth information
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box
            # focal_len = 150
            if labels[classId - 1] == 'couch':
                print(f"FOCAL LEN: {(w*220)/145}")
                # dist = (145*focal_len)/w
                # print(f"COUCH DIST: {dist} cm")
            # depth_values = depth_map[y:y+h, x:x+w]
            # avg_depth = np.mean(depth_values)
            # label = f'{labels[classId - 1]}: {confidence:.2f}, Depth: {avg_depth:.2f}m'
            label = f'{labels[classId - 1]}: {confidence:.2f}'
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # print(f"Object: {labels[classId - 1]}, Confidence: {confidence:.2f}, Avg Depth: {avg_depth:.2f}")

    return frame

def main():
    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Skip frames to improve performance
        if frame_count % skip_frames != 0:
            continue

        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        processed_frame = process_frame(frame_resized)

        cv2.imshow('Object Detection and Depth Estimation', processed_frame)

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}")

        time_to_wait = max(0, frame_interval - elapsed_time)
        time.sleep(time_to_wait)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()