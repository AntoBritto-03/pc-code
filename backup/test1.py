import cv2
import serial
import time
from ultralytics import YOLO

# ✅ Load the Fire & Smoke YOLOv8 model
fire_model = YOLO("models/fire_smoke_model.pt")

# ✅ Path to your video file
video_path = "videos/fire/fire_smoke7.mp4"
cap = cv2.VideoCapture(video_path)

# ✅ Connect to Arduino safely
try:
    arduino = serial.Serial('COM23', 115200, timeout=1)
    time.sleep(2)  # Wait for Arduino initialization
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ✅ Detection threshold
FIRE_CONFIDENCE_THRESHOLD = 0.4

# ✅ Set capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    fire_detected = False

    # ✅ Fire and Smoke Detection
    fire_results = fire_model(frame)[0]

    for box in fire_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())

        if confidence > FIRE_CONFIDENCE_THRESHOLD:
            fire_detected = True
            color = (0, 0, 255)  # Red for fire/smoke
            label = f"{fire_model.names[cls]} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Trigger Arduino alert
    if fire_detected:
        arduino.write(b'1')  # Send '1' for fire detection
        time.sleep(0.1)  # Prevent spamming
        print("🔥 Fire detected! Buzzer activated.")

    # ✅ Display the output
    cv2.imshow("Fire and Smoke Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Fire and Smoke Detection completed.")
