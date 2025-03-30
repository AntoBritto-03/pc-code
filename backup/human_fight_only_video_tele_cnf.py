import cv2
import serial
import time
import requests
from ultralytics import YOLO
import threading

# ✅ Telegram Bot Configuration
MAIN_BOT_TOKEN = "7697380823:AAGJxwH1eCM_jeK9xs82LpPfbqRLrgGFE5Y"   # Your bot token
MAIN_CHAT_ID = "1145760094"        # Your chat ID

# ✅ Police Bot Configuration
POLICE_BOT_TOKEN = "7821054209:AAF6wq-YflC10GktoBi36tTwlp1CdlljRvA"         # Replace with the police bot token
POLICE_CHAT_ID = "1145760094"             # Replace with the police chat ID

# ✅ YOLOv8 Model Configuration
violence_model = YOLO("models/human_fight_model.pt")

# ✅ Path to your video file
video_path = "videos/fighting/fighting1.mp4"  # Change to your fight video path
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

# ✅ Time control variables
LAST_SENT_TIME = 0  # To track the last frame sent timestamp
SEND_INTERVAL = 10  # 10 seconds interval
CONFIRMATION_RECEIVED = False

def send_telegram_photo(bot_token, chat_id, image_path, caption):
    """ Send the image with bounding box to Telegram """
    global LAST_SENT_TIME

    current_time = time.time()

    # ✅ Send only if 10 seconds have passed since the last frame
    if current_time - LAST_SENT_TIME >= SEND_INTERVAL:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(image_path, "rb") as photo:
            response = requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": photo})
            
        if response.status_code == 200:
            print(f"✅ Frame sent successfully to chat {chat_id}.")
            LAST_SENT_TIME = current_time  # Update the last sent time
        else:
            print(f"❌ Failed to send frame. Status: {response.status_code}")
    else:
        print("⚠️ Waiting 10 seconds before sending another frame...")

def check_for_confirmation():
    """ Poll the main Telegram bot for confirmation message """
    global CONFIRMATION_RECEIVED

    print("🚦 Polling for confirmation...")
    
    # ✅ Fetch the latest messages
    url = f"https://api.telegram.org/bot{MAIN_BOT_TOKEN}/getUpdates"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if "result" in data:
            for message in reversed(data["result"]):
                if "message" in message and "text" in message["message"]:
                    text = message["message"]["text"].lower()
                    chat_id = str(message["message"]["chat"]["id"])

                    # ✅ If the message contains "confirm" and comes from your chat ID
                    if text == "C" and chat_id == MAIN_CHAT_ID:
                        print("✅ Confirmation received! Sending frame to police.")
                        CONFIRMATION_RECEIVED = True
                        return
    else:
        print(f"❌ Failed to fetch messages. Status: {response.status_code}")

def start_polling():
    """ Start polling for confirmation in a separate thread """
    while True:
        check_for_confirmation()
        time.sleep(2)  # Poll every 2 seconds

# ✅ Start polling in a separate thread
polling_thread = threading.Thread(target=start_polling)
polling_thread.daemon = True
polling_thread.start()

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

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ✅ Save the frame locally only for violence
            image_path = "detected_violence.jpg"
            cv2.imwrite(image_path, frame)

        else:
            color = (0, 255, 0)  # Green for non-violence

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Trigger Arduino alert only for 'violence' class
    if violence_detected:
        arduino.write(b'2')  # Send '2' for violence detection
        time.sleep(0.1)  # Prevent spamming
        print("🚨 Violence detected! Buzzer activated.")
        
        # ✅ Send the frame to your main Telegram bot
        send_telegram_photo(MAIN_BOT_TOKEN, MAIN_CHAT_ID, "detected_violence.jpg", "🚨 Violence detected!")

    # ✅ If user confirms, send the frame to the police Telegram bot
    if CONFIRMATION_RECEIVED:
        send_telegram_photo(POLICE_BOT_TOKEN, POLICE_CHAT_ID, "detected_violence.jpg", "🚓 Violence confirmed! Sent to police.")
        CONFIRMATION_RECEIVED = False  # Reset confirmation flag

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
