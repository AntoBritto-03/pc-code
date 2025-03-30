import cv2
import serial
import time
import requests
from ultralytics import YOLO
import threading

# ✅ Telegram Bot Configuration
MAIN_BOT_TOKEN = "7556328060:AAHvtDPsjhPDPs0LSB8w1WLGaVXRKwfzxes"   # Main bot token
MAIN_CHAT_ID = "1516434632"        # Main chat ID

POLICE_BOT_TOKEN = "7719669178:AAEy2dWftDlgo-qV4tUTzl0C9YYvIo6Ep7w"         # Police bot token
POLICE_CHAT_ID = "1768043264"             # Police chat ID

# ✅ YOLOv8 Model Configuration
violence_model = YOLO("models/human_fight_model.pt")

# ✅ Open the camera instead of a video file
cap = cv2.VideoCapture(1)  # 🎥 Camera input (0 = default camera)

# ✅ Manually entered GPS coordinates
LATITUDE = "8.7321"   # Replace with your desired latitude
LONGITUDE = "77.7241"  # Replace with your desired longitude

# ✅ Function to get the manually entered GPS location
def get_manual_gps_location():
    return f"📍 Location: https://www.google.com/maps?q={LATITUDE},{LONGITUDE}"

# ✅ Connect to Arduino safely
try:
    arduino = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Wait for Arduino initialization
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ✅ Detection threshold
VIOLENCE_CONFIDENCE_THRESHOLD = 0.30

# ✅ Time control variables
LAST_SENT_TIME = 0  
SEND_INTERVAL = 10  
CONFIRMATION_RECEIVED = False
last_update_id = 0  # Track the last processed message ID
lock = threading.Lock()  # Prevent race conditions

# ✅ Function to send a photo to Telegram
def send_telegram_photo(bot_token, chat_id, image_path, caption):
    global LAST_SENT_TIME

    current_time = time.time()

    loc = get_manual_gps_location()
    caption = caption + loc

    if current_time - LAST_SENT_TIME >= SEND_INTERVAL:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(image_path, "rb") as photo:
            response = requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": photo})

        if response.status_code == 200:
            print(f"✅ Frame sent successfully to chat {chat_id}.")
            LAST_SENT_TIME = current_time
        else:
            print(f"❌ Failed to send frame. Status: {response.status_code}")
    else:
        print("⚠️ Waiting 10 seconds before sending another frame...")

# ✅ Function to send a police alert immediately
def send_police_alert():
    """ Send the frame to the police bot when confirmed """
    global CONFIRMATION_RECEIVED

    with lock:
        if CONFIRMATION_RECEIVED:
            print("🚓 Sending violence alert to police...")
            send_telegram_photo(POLICE_BOT_TOKEN, POLICE_CHAT_ID, "detected_violence.jpg", "🚓 Violence confirmed! Sent to police.")
            CONFIRMATION_RECEIVED = False  # Reset flag after sending

# ✅ Function to check for confirmation messages
def check_for_confirmation():
    """ Poll the main bot for confirmation """
    global CONFIRMATION_RECEIVED, last_update_id

    url = f"https://api.telegram.org/bot{MAIN_BOT_TOKEN}/getUpdates?offset={last_update_id + 1}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            if "result" in data:
                for message in data["result"]:
                    last_update_id = message["update_id"]

                    if "message" in message and "text" in message["message"]:
                        text = message["message"]["text"].lower()
                        chat_id = str(message["message"]["chat"]["id"])

                        if text == "c" and chat_id == MAIN_CHAT_ID:
                            print("✅ Confirmation received!")
                            with lock:
                                CONFIRMATION_RECEIVED = True
                            send_police_alert()
                            return
        else:
            print(f"❌ Failed to fetch messages. Status: {response.status_code}")

    except Exception as e:
        print(f"❌ Error polling confirmation: {e}")

# ✅ Start polling in a separate thread
def start_polling():
    while True:
        check_for_confirmation()
        time.sleep(2)

# ✅ Launch polling thread
polling_thread = threading.Thread(target=start_polling)
polling_thread.daemon = True
polling_thread.start()

# ✅ Main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected or error reading frame.")
        break

    frame = cv2.resize(frame, (600, 600))
    violence_detected = False

    # ✅ YOLOv8 Detection
    violence_results = violence_model(frame)[0]

    for box in violence_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())
        class_name = violence_model.names[cls].lower()
        label = f"{class_name} ({confidence:.2f})"

        if class_name == "violence" and confidence > VIOLENCE_CONFIDENCE_THRESHOLD:
            violence_detected = True
            color = (0, 0, 255)  

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the frame
            image_path = "detected_violence.jpg"
            cv2.imwrite(image_path, frame)

        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Trigger Arduino buzzer for violence
    if violence_detected:
        arduino.write(b'2')
        time.sleep(0.1)
        print("🚨 Violence detected! Buzzer activated.")
        
        # ✅ Send image to main bot
        send_telegram_photo(MAIN_BOT_TOKEN, MAIN_CHAT_ID, "detected_violence.jpg", "🚨 Violence detected!")

    # ✅ Display the frame
    cv2.imshow("Human Fight (Violence) Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()

print("Human Fight Detection completed.")
