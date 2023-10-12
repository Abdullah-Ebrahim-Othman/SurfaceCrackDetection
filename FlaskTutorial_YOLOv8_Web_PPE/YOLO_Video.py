from ultralytics import YOLO
import cv2
import math

import time


def video_detection(path_x, fps=10):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Set the frame rate (FPS) delay
    frame_delay = 1 / fps

    # Load the YOLO model
    model = YOLO("C:\\Users\\Abdallh\\Desktop\\YOLOv8-CrashCourse\\Yolo Weights\\300 epochs\\best.pt")

    classNames = ['concrete-crack']

    while True:
        success, img = cap.read()

        if not success:
            break  # Exit the loop if the video is over

        # Perform object detection
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]

                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                if class_name == 'concrete-crack':
                    color = (0, 204, 255)
                else:
                    color = (85, 45, 255)

                if conf > 0.4:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img
        time.sleep(frame_delay)  # Introduce frame rate delay


cv2.destroyAllWindows()  # Close any open windows when done

