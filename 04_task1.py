from ultralytics import YOLO
import cv2
from collections import deque
import time

model = YOLO("yolo11n-pose.pt")

wrist_x_positions = deque(maxlen=10)
wave_display_end = 0

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    for keypoint_data in results[0].keypoints.data:
        right_wrist = keypoint_data[10]
        if right_wrist is not None and right_wrist[2] > 0.5:
            cv2.circle(annotated_frame, 
                       (int(right_wrist[0]), int(right_wrist[1])), 
                       10, (0, 0, 255), -1)
            wrist_x = int(right_wrist[0])
            wrist_x_positions.append(wrist_x)
            if len(wrist_x_positions) >= 10:
                min_x = min(wrist_x_positions)
                max_x = max(wrist_x_positions)
                movement = max_x - min_x

                if movement > 500 and current_time > wave_display_end:
                    wave_display_end = current_time + 2

    if current_time < wave_display_end:
        cv2.putText(annotated_frame, "Wave Detected!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Annotated Video Stream', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()