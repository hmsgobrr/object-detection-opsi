import cv2
import time
import numpy as np
import tensorflow as tf

# Load COCO labels
with open('coco.names') as f:
    labels = f.read().rstrip('\n').split('\n')

# Initialize Object Detection model (MobileNet SSD using TensorFlow Lite)
model_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
interpreter = tf.lite.Interpreter(model_path=model_path,
                                  experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')])
interpreter.allocate_tensors()

# Get input and output tensors for object detection
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load TinyDepth model for depth estimation
depth_model_path = "1.tflite"
depth_interpreter = tf.lite.Interpreter(model_path=depth_model_path,
                                        experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')])
depth_interpreter.allocate_tensors()

depth_input_details = depth_interpreter.get_input_details()
depth_output_details = depth_interpreter.get_output_details()

# Initialize camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_width = 120
frame_height = 90
thres = 0.60
target_fps = 10
frame_interval = 1.0 / target_fps
skip_frames = 2  # Skip every 2nd frame for efficiency

def preprocess_image(image, input_shape):
    """Preprocess the image for the model."""
    img = cv2.resize(image, (input_shape[1], input_shape[2]))  # Resize to the model's input size
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize and add batch dimension
    return img

def process_frame(frame):
    # Preprocess the frame for object detection
    input_image = preprocess_image(frame, input_details[0]['shape'])

    # Object detection inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Preprocess the frame for depth estimation
    depth_input_image = preprocess_image(frame, depth_input_details[0]['shape'])

    # Depth estimation inference
    depth_interpreter.set_tensor(depth_input_details[0]['index'], depth_input_image)
    depth_interpreter.invoke()

    # Get depth map output
    depth_map = depth_interpreter.get_tensor(depth_output_details[0]['index'])
    depth_map = depth_map.squeeze()  # Remove unnecessary dimensions
    depth_map = cv2.resize(depth_map, (frame_width, frame_height))  # Resize to match the original frame size

    # Draw bounding boxes and depth information
    for i in range(len(scores[0])):
        if scores[0][i] > thres:
            class_id = int(classes[0][i])
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = box
            x, y, w, h = int(xmin * frame_width), int(ymin * frame_height), int((xmax - xmin) * frame_width), int((ymax - ymin) * frame_height)
            
            depth_values = depth_map[y:y+h, x:x+w]
            avg_depth = np.mean(depth_values)
            label = f'{labels[class_id]}: {scores[0][i]:.2f}, Depth: {avg_depth:.2f}m'
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Object: {labels[class_id]}, Confidence: {scores[0][i]:.2f}, Avg Depth: {avg_depth:.2f}")

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

        # cv2.imshow('Object Detection and Depth Estimation', processed_frame)

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}")

        time_to_wait = max(0, frame_interval - elapsed_time)
        time.sleep(time_to_wait)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()