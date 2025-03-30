import cv2
import torch
import serial
import time
import requests
from datetime import datetime
from ultralytics import YOLO

# ✅ Load the YOLOv8 model
model = YOLO("models/human_model.pt")  # Path to your human detection model

# ✅ Use USB webcam (index 0 for default camera)
cap = cv2.VideoCapture(1)

# ✅ Connect to Arduino
try:
    arduino = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Allow Arduino initialization
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

# ✅ Telegram Bot Info
BOT_TOKEN = "7697380823:AAGJxwH1eCM_jeK9xs82LpPfbqRLrgGFE5Y"
CHAT_ID = "1145760094"

# ✅ Set the start and end time (24-hour format)
START_TIME = "8:3"   # Change this to your desired start time
END_TIME = "8:10"     # Change this to your desired end time

# ✅ Rate-limiting variables
ALERT_INTERVAL = 20  # Minimum 10 seconds between alerts
last_alert_time = 0  # Timestamp of the last Telegram alert

# ✅ Function to check if the current time is within the range
def is_within_time_range(start, end):
    """ Check if current time is between start and end time """
    now = datetime.now().time()
    start_time = datetime.strptime(start, "%H:%M").time()
    end_time = datetime.strptime(end, "%H:%M").time()

    if start_time < end_time:
        return start_time <= now <= end_time
    else:
        return now >= start_time or now <= end_time

# ✅ Function to send a frame image to Telegram
def send_telegram_frame(image_path, message="🚨 Human detected!"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    
    with open(image_path, "rb") as image:
        files = {"photo": image}
        data = {"chat_id": CHAT_ID, "caption": message}
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        print("✅ Frame sent successfully to Telegram!")
    else:
        print(f"❌ Failed to send frame: {response.text}")

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# ✅ Main Loop
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam.")
        break

    # ✅ Resize the frame
    frame = cv2.resize(frame, (600, 500))

    # ✅ YOLO detection
    results = model(frame)

    human_detected = False

    # ✅ Process the detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())

        label = f"Person ({confidence:.2f})"
        color = (0, 255, 0)  # Green for humans

        # ✅ Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ✅ Detect human with confidence > 0.4
        if confidence > 0.5:
            human_detected = True

    # ✅ Send buzzer signal and Telegram frame only if within the time range
    current_time = time.time()

    if human_detected and is_within_time_range(START_TIME, END_TIME):
        arduino.write(b'3')  # Send '3' to activate the buzzer
        print(f"🚨 Human detected! Buzzer activated at {datetime.now().strftime('%H:%M:%S')}")

        # ✅ Only send Telegram alert every 10 seconds
        if current_time - last_alert_time >= ALERT_INTERVAL:
            # ✅ Save the detected frame as an image
            frame_path = f"detected_frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            # ✅ Send the frame to Telegram
            send_telegram_frame(frame_path)

            # ✅ Remove the temporary image file
            import os
            os.remove(frame_path)

            # ✅ Update the last alert timestamp
            last_alert_time = current_time
            frame_count += 1

    # ✅ Display the frame
    cv2.imshow("Human Detection", frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Human Detection completed.")
