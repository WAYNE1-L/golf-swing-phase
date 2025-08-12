
import cv2
import pandas as pd
import tkinter as tk
from tkinter import ttk

# === CONFIG ===
VIDEO_PATH = "training_data/swing5.mp4"
CSV_PATH = "keypoints_csv/swing5.csv"
OUTPUT_CSV_PATH = "labeled_csv/swing5_labeled.csv"






# === Load video and keypoints ===
df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# === Phases ===
PHASES = ["Setup", "Takeaway", "Backswing", "Downswing", "Impact", "Follow-through"]

# === GUI ===
current_frame = 0
labels = [""] * len(df)

def update_frame(idx):
    global current_frame
    current_frame = max(0, min(len(df)-1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Swing Frame", frame)
    cv2.waitKey(1)
    phase_label.set(labels[current_frame] if labels[current_frame] else "Unlabeled")
    frame_label.config(text=f"Frame {current_frame+1} / {len(df)}")

def mark_phase(phase):
    labels[current_frame] = phase
    update_frame(current_frame + 1)

def save_labels():
    df["phase"] = labels
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    status_label.config(text="✅ Saved to " + OUTPUT_CSV_PATH)

root = tk.Tk()
root.title("Swing Phase Labeler")

frame_label = ttk.Label(root, text="")
frame_label.pack()

phase_label = tk.StringVar()
ttk.Label(root, textvariable=phase_label).pack()

btn_frame = ttk.Frame(root)
btn_frame.pack()

for phase in PHASES:
    ttk.Button(btn_frame, text=phase, command=lambda p=phase: mark_phase(p)).pack(side=tk.LEFT, padx=4)

nav_frame = ttk.Frame(root)
nav_frame.pack(pady=10)

ttk.Button(nav_frame, text="<< Prev", command=lambda: update_frame(current_frame - 1)).pack(side=tk.LEFT, padx=4)
ttk.Button(nav_frame, text="Next >>", command=lambda: update_frame(current_frame + 1)).pack(side=tk.LEFT, padx=4)
ttk.Button(nav_frame, text="💾 Save", command=save_labels).pack(side=tk.LEFT, padx=4)

status_label = ttk.Label(root, text="")
status_label.pack(pady=5)

update_frame(0)
root.mainloop()
