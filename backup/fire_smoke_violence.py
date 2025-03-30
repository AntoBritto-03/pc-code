import cv2
import torch
import serial
import time
from ultralytics import YOLO

# ✅ Load the YOLOv8 models
fire_model = YOLO("models/fire_smoke_model.pt")  # Fire & Smoke detection model
violence_model = YOLO("models/human_fight_model.pt")  # Human violence detection model

# ✅ Path to your video file
video_path = "videos/fire/fire_smoke7.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# ✅ Connect to Arduino (Change 'COM3' to your actual port)
arduino = serial.Serial('COM23', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ✅ Detection thresholds
FIRE_CONFIDENCE_THRESHOLD = 0.4
VIOLENCE_CONFIDENCE_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # ✅ Resize the frame to 600x500
    frame = cv2.resize(frame, (600, 500))

    # ✅ Fire and Smoke Detection
    fire_results = fire_model(frame)

    fire_detected = False
    violence_detected = False

    # ✅ Process Fire and Smoke Detection
    for result in fire_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            label = f"{fire_model.names[cls]} ({confidence:.2f})"
            color = (0, 0, 255)  # Red for fire/smoke

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Fire alert condition
            if confidence > FIRE_CONFIDENCE_THRESHOLD:
                fire_detected = True

    # ✅ Human and Violence Detection
    violence_results = violence_model(frame)

    # ✅ Process Violence Detection
    for result in violence_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Determine if human or violence
            label = f"{violence_model.names[cls]} ({confidence:.2f})"
            
            # Color coding
            if "violence" in violence_model.names[cls].lower():
                color = (0, 0, 255)  # Red for violence
                violence_detected = True
            else:
                color = (0, 255, 0)  # Green for normal humans

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Trigger Arduino alerts based on detections
    if fire_detected:
        arduino.write(b'1')  # Send '1' for fire detection
        print("🔥 Fire detected! Buzzer activated.")

    if violence_detected:
        arduino.write(b'2')  # Send '2' for violence detection
        print("🚨 Violence detected! Buzzer activated.")

    # ✅ Display the combined output
    cv2.imshow("Fire, Smoke, and Violence Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Detection completed.")
