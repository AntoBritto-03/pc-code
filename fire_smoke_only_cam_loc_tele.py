import cv2
import serial
import time
from datetime import datetime
import requests
from ultralytics import YOLO

# ✅ Load the Fire & Smoke YOLOv8 model
fire_model = YOLO("models/fire_smoke_model.pt")

# ✅ Use camera input (index 0 for the default webcam)
cap = cv2.VideoCapture(1)

# ✅ Connect to Arduino safely
try:
    arduino = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Wait for Arduino initialization
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

if not cap.isOpened():
    print("Error: Could not access camera.")
    exit()

# ✅ Detection threshold
FIRE_CONFIDENCE_THRESHOLD = 0.40

# ✅ Telegram Bot Configuration
TELEGRAM_BOT_1_TOKEN = "7697380823:AAGJxwH1eCM_jeK9xs82LpPfbqRLrgGFE5Y"  # First bot token
TELEGRAM_BOT_2_TOKEN = "7821054209:AAF6wq-YflC10GktoBi36tTwlp1CdlljRvA"  # Second bot token (Replace with actual token)
CHAT_ID_1 = "1145760094"  # First bot chat ID
CHAT_ID_2 = "1145760094"  # Second bot chat ID (Replace with actual chat ID)
last_telegram_alert = 0  # Timestamp for last Telegram message
TELEGRAM_INTERVAL = 10  # Send message every 10 seconds

# ✅ Manually entered GPS coordinates
LATITUDE = "12.955939"   # Replace with your desired latitude
LONGITUDE = "80.246234"  # Replace with your desired longitude

# ✅ Function to get the manually entered GPS location
def get_manual_gps_location():
    return f"📍 Location: https://www.google.com/maps?q={LATITUDE},{LONGITUDE}"

# ✅ Function to send Telegram message
def send_telegram_alert(bot_token, chat_id, message):
    telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(telegram_url, json=payload)
        if response.status_code == 200:
            print(f"✅ Telegram alert sent to bot with chat ID {chat_id}.")
        else:
            print(f"⚠️ Failed to send Telegram alert to bot {chat_id}. Status code: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram send error for bot {chat_id}: {e}")

# ✅ Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from camera.")
        break

    # ✅ Resize the frame to 600x600 for display
    frame = cv2.resize(frame, (600, 600))

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

        # ✅ Send Telegram alert every 10 seconds with manual GPS location
        current_time = time.time()

        if current_time - last_telegram_alert >= TELEGRAM_INTERVAL:
            location = get_manual_gps_location()  # 🔥 Use the manual GPS location
            message = f"🔥Fire/Smoke Detected🔥\n{location}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # ✅ Send alert to both Telegram bots
            send_telegram_alert(TELEGRAM_BOT_1_TOKEN, CHAT_ID_1, message)
            send_telegram_alert(TELEGRAM_BOT_2_TOKEN, CHAT_ID_2, message)

            last_telegram_alert = current_time

    # ✅ Display the resized frame
    cv2.imshow("Fire and Smoke Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Fire and Smoke Detection completed.")
