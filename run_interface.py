import subprocess
import tkinter as tk

# Store process references
process1 = None
process2 = None

# Function to start first script
def start_fire_smoke():
    global process1
    stop_all()
    process1 = subprocess.Popen(["python", "fire_smoke_only_cam_loc_tele.py"])
    status_label.config(text="🔥 Running: Fire & Smoke Detection")

# Function to start second script
def start_human_fight():
    stop_all()
    global process2
    process2 = subprocess.Popen(["python", "human_fight_only_cam_tele_cnf.py"])  # Change to actual file
    status_label.config(text="🕵️ Running: Motion Detection")

# Function to stop first script
def stop_fire_smoke():
    global process1
    if process1:
        process1.terminate()
        process1 = None
        status_label.config(text="✅ Stopped Fire & Smoke Detection")

# Function to stop second script
def stop_human_fight():
    global process2
    if process2:
        process2.terminate()
        process2 = None
        status_label.config(text="✅ Stopped Motion Detection")


def stop_all():
    stop_fire_smoke()
    stop_human_fight()
    status_label.config(text="✅ All scripts stopped")

# Create GUI window
root = tk.Tk()
root.title("AI Surveillance Control")
root.geometry("400x300")

# Buttons
btn_start_fire = tk.Button(root, text="Start Fire & Smoke Detection", command=start_fire_smoke, bg="red", fg="white", width=30)
btn_start_fire.pack(pady=10)

btn_start_human_fight = tk.Button(root, text="Start Motion Detection", command=start_human_fight, bg="blue", fg="white", width=30)
btn_start_human_fight.pack(pady=10)

btn_stop_fire = tk.Button(root, text="Stop Fire & Smoke", command=stop_fire_smoke, bg="gray", fg="white", width=30)
btn_stop_fire.pack(pady=5)

btn_stop_human_fight = tk.Button(root, text="Stop Motion Detection", command=stop_human_fight, bg="gray", fg="white", width=30)
btn_stop_human_fight.pack(pady=5)

# Status Label
status_label = tk.Label(root, text="Idle", font=("Arial", 12), fg="green")
status_label.pack(pady=20)

# Run GUI
root.mainloop()
