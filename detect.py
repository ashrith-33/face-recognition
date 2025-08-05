"""
import face_recognition
import cv2
import os
import pickle
import numpy as np
import time
import sys
ENCODINGS_FILE = "trained_faces.pkl"

INPUT_SOURCE = 'webcam' 

#INPUT_SOURCE = 'image'
#INPUT_SOURCE = 'video'
#IMAGE_PATH = "test_image.jpg" 
#VIDEO_PATH = "test_video.mp4" 
#A higher value means less frequent detection, but smoother video.
FRAME_SKIP_INTERVAL = 1 # Adjusted for smoother webcam feed
#Tolerance for face comparison. Lower value means stricter match.
RECOGNITION_TOLERANCE = 0.4
def load_encodings(file_path):
    print(f"[INFO] Loading encodings from {file_path}...")
    if not os.path.exists(file_path):
        print(f"[ERROR] Encoded faces file '{file_path}' not found. "
              "Please run 'train.py' first to generate it.")
        return [], []
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    except Exception as e:
        print(f"[ERROR] Could not load encodings from {file_path}: {e}")
        return [], []

def process_frame(frame, known_face_encodings, known_face_names, frame_number, process_this_frame, current_fps):
    face_locations = []
    face_encodings = []
    face_names = []

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    else:
        pass


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2) 

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1) 

    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def run_detection():

    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

    if not known_face_encodings:
        print("[ERROR] Exiting: No known faces available for detection.")
        sys.exit(1)

    video_capture = None
    if INPUT_SOURCE == 'webcam':
        video_capture = cv2.VideoCapture(0) 
        if not video_capture.isOpened():
            print("[ERROR] Could not open webcam. Exiting.")
            print("Please check:")
            print("  1. If your webcam is connected and working.")
            print("  2. If other applications are using the webcam.")
            print("  3. Your system's privacy settings for camera access.")
            print("  4. Try changing cv2.VideoCapture(0) to (1), (2), etc., if you have multiple cameras.")
            sys.exit(1)
        print("[INFO] Starting webcam feed. Press 'q' to quit.")
        print("IMPORTANT: Click on the OpenCV window to make it active before pressing 'q' to quit.")
    elif INPUT_SOURCE == 'video':
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERROR] Video file '{VIDEO_PATH}' not found. Exiting.")
            sys.exit(1)
        video_capture = cv2.VideoCapture(VIDEO_PATH)
        if not video_capture.isOpened():
            print(f"[ERROR] Could not open video file '{VIDEO_PATH}'. Exiting.")
            sys.exit(1)
        print(f"[INFO] Processing video: {VIDEO_PATH}. Press 'q' to quit.")
        print("IMPORTANT: Click on the OpenCV window to make it active before pressing 'q' to quit.")
    elif INPUT_SOURCE == 'image':
        if not os.path.exists(IMAGE_PATH):
            print(f"[ERROR] Image file '{IMAGE_PATH}' not found. Exiting.")
            sys.exit(1)
        print(f"[INFO] Processing image: {IMAGE_PATH}.")
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print(f"[ERROR] Could not load image '{IMAGE_PATH}'. Exiting.")
            sys.exit(1)
        processed_image = process_frame(image, known_face_encodings, known_face_names, 0, True, 0.0) # FPS not relevant for static image
        cv2.imshow('Person Detection Result', processed_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return 
    else:
        print(f"[ERROR] Invalid INPUT_SOURCE: '{INPUT_SOURCE}'. Choose 'image', 'video', or 'webcam'.")
        sys.exit(1)


    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] End of video stream or failed to read frame.")
            break

        frame_count += 1
        process_this_frame = (frame_count % FRAME_SKIP_INTERVAL == 0)

        processed_frame = process_frame(frame, known_face_encodings, known_face_names, frame_count, process_this_frame, fps)

        cv2.imshow('Video/Webcam Feed - Person Detection', processed_frame)

        if frame_count % 10 == 0: 
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed. Exiting.")
            break


    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection process finished.")

if __name__ == "__main__":
    run_detection()
"""
# -*- coding: utf-8 -*-
"""
This script performs real-time person detection, logs known entries,
and captures images of unknown individuals.

Features:
- Logs entry time of recognized individuals only once per cooldown period.
- Saves snapshots of unknown individuals only once per cooldown period.
- Implements cooldowns to prevent spamming logs/captures.
- Optimized for smoother real-time performance.
"""

import face_recognition
import cv2
import os
import pickle
import numpy as np
import time
import sys
from datetime import datetime

# --- Configuration ---
ENCODINGS_FILE = "trained_faces.pkl"

# IMPORTANT: Only ONE of the INPUT_SOURCE lines below should be uncommented.
INPUT_SOURCE = 'webcam'

IMAGE_PATH = "test_image.jpg"
# VIDEO_PATH = "test_video.mp4"

# Adjusted for smoother, event-based detection
FRAME_SKIP_INTERVAL = 5
RECOGNITION_TOLERANCE = 0.4

# --- New Configuration for Logging and Capturing ---
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "detection_log.txt")
UNKNOWN_CAPTURES_DIR = "unknown_captures"

# Cooldown period in seconds for logging/capturing the same person/unknown face
LOG_COOLDOWN_SECONDS = 30
UNKNOWN_CAPTURE_COOLDOWN_SECONDS = 10

# Dictionaries to store the last time a person was logged/captured
last_logged_time = {} # Format: {person_name: timestamp}
last_unknown_capture = {'encoding': None, 'timestamp': 0} # Stores info about the last captured unknown face

# A simple counter for unique unknown faces in a session (resets on script restart)
unknown_face_counter = 0


def ensure_directories_exist():
    """Ensures that the log and unknown captures directories exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_CAPTURES_DIR, exist_ok=True)
    print(f"[INFO] Ensured '{LOG_DIR}' and '{UNKNOWN_CAPTURES_DIR}' directories exist.")

def log_detection(person_name):
    """Logs the detection of a known person with a timestamp."""
    current_time = time.time()
    if person_name not in last_logged_time or \
       (current_time - last_logged_time[person_name]) > LOG_COOLDOWN_SECONDS:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Recognized: {person_name}\n"
        try:
            with open(LOG_FILE, "a") as f:
                f.write(log_entry)
            print(f"[LOG] {log_entry.strip()}")
            last_logged_time[person_name] = current_time
        except Exception as e:
            print(f"[ERROR] Could not write to log file: {e}")

def capture_unknown_face(frame, face_encoding, face_location):
    """
    Captures and saves an image of an unknown face only once per cooldown period.
    """
    global unknown_face_counter
    global last_unknown_capture

    current_time = time.time()

    # Check if a similar unknown face has been captured recently
    if last_unknown_capture['encoding'] is not None:
        face_distances = face_recognition.face_distance([last_unknown_capture['encoding']], face_encoding)
        # If the new unknown face is a close match to the last captured one,
        # and is within the cooldown period, do not capture again.
        if face_distances[0] < RECOGNITION_TOLERANCE and \
           (current_time - last_unknown_capture['timestamp']) < UNKNOWN_CAPTURE_COOLDOWN_SECONDS:
            return

    # Extract the face region from the frame
    top, right, bottom, left = face_location
    margin = 20
    top = max(0, top - margin)
    right = min(frame.shape[1], right + margin)
    bottom = min(frame.shape[0], bottom + margin)
    left = max(0, left - margin)
    face_image = frame[top:bottom, left:right]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(UNKNOWN_CAPTURES_DIR, f"unknown_{timestamp_str}_{unknown_face_counter}.jpg")
    try:
        cv2.imwrite(filename, face_image)
        print(f"[CAPTURE] Captured unknown face to: {filename}")
        last_unknown_capture['encoding'] = face_encoding
        last_unknown_capture['timestamp'] = current_time
        unknown_face_counter += 1
    except Exception as e:
        print(f"[ERROR] Could not save unknown face image: {e}")


def load_encodings(file_path):
    """Loads known face encodings and names from a pickle file."""
    print(f"[INFO] Loading encodings from {file_path}...")
    if not os.path.exists(file_path):
        print(f"[ERROR] Encoded faces file '{file_path}' not found. "
              "Please run 'train.py' first to generate it.")
        return [], []
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    except Exception as e:
        print(f"[ERROR] Could not load encodings from {file_path}: {e}")
        return [], []

def process_frame(frame, known_face_encodings, known_face_names, frame_number, process_this_frame, current_fps):
    face_locations = []
    face_encodings = []
    face_names = []
    
    # Store previous face locations to simulate 'moving towards'
    global last_known_locations
    current_known_locations = {}

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                log_detection(name)
            else:
                top, right, bottom, left = face_locations[i]
                original_face_location = (top * 4, right * 4, bottom * 4, left * 4)
                capture_unknown_face(frame, face_encoding, original_face_location)

            face_names.append(name)
            current_known_locations[name] = face_locations[i]
    else:
        # Simple placeholder for event logic. For production, you'd use a tracker here.
        pass

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def run_detection():
    ensure_directories_exist()

    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

    if not known_face_encodings:
        print("[ERROR] Exiting: No known faces available for detection.")
        sys.exit(1)

    if 'INPUT_SOURCE' not in globals():
        print("[ERROR] INPUT_SOURCE is not defined. Please uncomment one of the INPUT_SOURCE options in the configuration section.")
        sys.exit(1)

    if INPUT_SOURCE == 'webcam':
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("[ERROR] Could not open webcam. Exiting.")
            print("Please check:")
            print("  1. If your webcam is connected and working.")
            print("  2. If other applications are using the webcam.")
            print("  3. Your system's privacy settings for camera access.")
            print("  4. Try changing cv2.VideoCapture(0) to (1), (2), etc., if you have multiple cameras.")
            sys.exit(1)
        print("[INFO] Starting webcam feed. Press 'q' to quit.")
        print("IMPORTANT: Click on the OpenCV window to make it active before pressing 'q' to quit.")
    elif INPUT_SOURCE == 'video':
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERROR] Video file '{VIDEO_PATH}' not found. Exiting.")
            sys.exit(1)
        video_capture = cv2.VideoCapture(VIDEO_PATH)
        if not video_capture.isOpened():
            print(f"[ERROR] Could not open video file '{VIDEO_PATH}'. Exiting.")
            sys.exit(1)
        print(f"[INFO] Processing video: {VIDEO_PATH}. Press 'q' to quit.")
        print("IMPORTANT: Click on the OpenCV window to make it active before pressing 'q' to quit.")
    elif INPUT_SOURCE == 'image':
        if not os.path.exists(IMAGE_PATH):
            print(f"[ERROR] Image file '{IMAGE_PATH}' not found. Exiting.")
            sys.exit(1)
        print(f"[INFO] Processing image: {IMAGE_PATH}.")
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print(f"[ERROR] Could not load image '{IMAGE_PATH}'. Exiting.")
            sys.exit(1)
        processed_image = process_frame(image, known_face_encodings, known_face_names, 0, True, 0.0)
        cv2.imshow('Person Detection Result', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        print(f"[ERROR] Invalid INPUT_SOURCE: '{INPUT_SOURCE}'. Choose 'image', 'video', or 'webcam'.")
        sys.exit(1)

    frame_count = 0
    start_time = time.time()
    fps = 0.0
    last_known_locations = {} # Added to store locations for event-based logic

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] End of video stream or failed to read frame.")
            break

        frame_count += 1
        process_this_frame = (frame_count % FRAME_SKIP_INTERVAL == 0)

        processed_frame = process_frame(frame, known_face_encodings, known_face_names, frame_count, process_this_frame, fps)

        cv2.imshow('Video/Webcam Feed - Person Detection', processed_frame)

        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed. Exiting.")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection process finished.")

if __name__ == "__main__":
    run_detection()
