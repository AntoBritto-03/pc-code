import cv2
import serial
import time
from ultralytics import YOLO

# ✅ Load the Human Fight Detection YOLOv8 model
violence_model = YOLO("models/human_fight_model.pt")

# ✅ Path to your video file
video_path = "videos/fighting/fighting2.mp4"  # Change to your fight video path
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
VIOLENCE_CONFIDENCE_THRESHOLD = 0.30

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # ✅ Resize the frame to 600x600 for display
    frame = cv2.resize(frame, (600, 600))

    violence_detected = False

    # ✅ Violence Detection
    violence_results = violence_model(frame)[0]

    for box in violence_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())

        # Get the class name
        class_name = violence_model.names[cls].lower()

        label = f"{class_name} ({confidence:.2f})"

        if class_name == "violence" and confidence > VIOLENCE_CONFIDENCE_THRESHOLD:
            violence_detected = True
            color = (0, 0, 255)  # Red for violence
        else:
            color = (0, 255, 0)  # Green for non-violence

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Trigger Arduino alert only for 'violence' class
    if violence_detected:
        arduino.write(b'2')  # Send '2' for violence detection
        time.sleep(0.1)  # Prevent spamming
        print("🚨 Violence detected! Buzzer activated.")

    # ✅ Display the resized frame
    cv2.imshow("Human Fight (Violence) Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Human Fight Detection completed.")
