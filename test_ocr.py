import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr

VIDEO_PATH = 'test.mp4'  
CAR_REAL_WIDTH = 1.8  # Assumed average width of a car in meters

model = YOLO('yolov8n.pt')  
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
tracker_dict = {}  # For storing previous centroids and sizes: {vehicle_id: [prev_cx, prev_cy, prev_w, prev_frame]}

def estimate_speed(prev_cx, prev_cy, cur_cx, cur_cy, prev_w, cur_w, fps):
    avg_w = (prev_w + cur_w) / 2
    meters_per_pixel = CAR_REAL_WIDTH / avg_w if avg_w != 0 else 0
    pixel_dist = np.sqrt((cur_cx - prev_cx)**2 + (cur_cy - prev_cy)**2)
    meters = pixel_dist * meters_per_pixel
    seconds = 1 / fps
    speed_m_s = meters / seconds if seconds != 0 else 0
    speed_kmh = speed_m_s * 3.6
    return speed_kmh

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended.")
        break

    frame_count += 1

    results = model(frame)
    current_ids = []

    # Speed Estimation
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(b) for b in box]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            box_w = x2 - x1
            vehicle_id = f"{x1}_{y1}"

            current_ids.append(vehicle_id)

            speed_text = ""
            if vehicle_id in tracker_dict:
                prev_cx, prev_cy, prev_w, prev_frame = tracker_dict[vehicle_id]
                speed = estimate_speed(prev_cx, prev_cy, cx, cy, prev_w, box_w, fps)
                speed_text = f"{speed:.1f} km/h"
                cv2.putText(frame, speed_text, (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            tracker_dict[vehicle_id] = [cx, cy, box_w, frame_count]

            # Number Plate Recognition
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size > 0:
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)
                ocr_results = reader.readtext(gray_plate)
                plate_text = ""
                if ocr_results:
                    plate_text = ocr_results[0][-2]
                    if len(plate_text) > 3 and any(c.isdigit() for c in plate_text):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        print(f"Frame {frame_count}: Plate detected - {plate_text} | Speed: {speed_text}")

    for old_id in list(tracker_dict):
        if old_id not in current_ids:
            del tracker_dict[old_id]

    cv2.imshow('NPR + Speed Estimation Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User exited.")
        break

cap.release()
cv2.destroyAllWindows()
