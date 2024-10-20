import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import numpy as np
import threading
import pygame
from tkinter import messagebox

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize pygame for playing beep sound
pygame.mixer.init()
# alert_sound = pygame.mixer.Sound('alert_beep.wav')  # Make sure to have a beep sound file

# # Function to play beep sound
# def play_beep():
#     pygame.mixer.Sound.play(alert_sound)

# Function to perform crowd analysis
def analyze_frame(frame):
    results = model(frame)  # YOLOv5 inference
    labels = results.xyxy[0][:, -1].cpu().numpy()  # Extract labels (class IDs)
    
    # Count number of people detected (YOLO class ID for person is 0)
    people_count = np.sum(labels == 0)
    
    # Define a crowd threshold (e.g., 5 people)
    if people_count > 5:
        # play_beep()
        print("overcrowded people")
    
    # Draw bounding boxes on frame
    results.render()  # Add bounding boxes to frame
    return frame

# Function to process video input
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = analyze_frame(frame)  # Analyze the frame
        
        # Display the frame
        cv2.imshow('Crowd Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle video upload
def upload_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if video_path:
        threading.Thread(target=process_video, args=(video_path,)).start()

# Function to connect to laptop camera
def connect_camera():
    camera_index = 0  # Default camera index
    threading.Thread(target=process_video, args=(camera_index,)).start()

# Function to connect via RTSP
def connect_rtsp():
    rtsp_url = rtsp_entry.get()
    if rtsp_url:
        threading.Thread(target=process_video, args=(rtsp_url,)).start()

# Create the main window
root = tk.Tk()
root.title("Crowd Analysis Application")

# Buttons and inputs
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=10)

camera_button = tk.Button(root, text="Connect Laptop Camera", command=connect_camera)
camera_button.pack(pady=10)

# RTSP input
rtsp_label = tk.Label(root, text="RTSP Camera URL:")
rtsp_label.pack(pady=10)
rtsp_entry = tk.Entry(root)
rtsp_entry.pack(pady=10)

rtsp_button = tk.Button(root, text="Connect RTSP Camera", command=connect_rtsp)
rtsp_button.pack(pady=10)

# Start the GUI loop
root.mainloop()
