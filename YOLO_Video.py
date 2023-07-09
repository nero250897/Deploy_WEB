from ultralytics import YOLO
import cv2
import math
import time
import numpy as np

def video_detection(path_x):
    video_capture = path_x

    #Tao webcam
    cap=cv2.VideoCapture(video_capture)

    model=YOLO("C:/Users/84335/PycharmProjects/Doan/Detect_Person/best.pt")
    classNames = ["person"]

    # Khoi tao bien de tinh toan FPS
    start_time = time.time()
    frame_count = 0

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:  # Vong lap hien thi bouding box cho moi ket qua
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Mau xanh

                # Tinh toan centroid
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(img, centroid, 5, (0, 255, 0), -1)  # Mau xanh la
                centroid_array = np.array(centroid)
                # print(centroid_array)

                # print(box.conf[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                # print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)  # filled Mau xanh la
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1,
                            lineType=cv2.LINE_AA)  # Mau trang

        # Tinh toan FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(img, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        yield img

cv2.destroyAllWindows()