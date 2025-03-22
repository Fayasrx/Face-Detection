import cv2
import time
import pygame
import threading

# Initialize pygame mixer for sound
pygame.mixer.init()

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Constants for sleep detection
FRAMES_THRESHOLD = 15  # Number of frames where eyes should be detected as "closed" before triggering sleep alert
ALARM_TRIGGERED = False  # Flag to ensure the alarm is played only once when triggered

# Initialize variables
frame_counter = 0
eyes_closed_frames = 0

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

# Function to play the alarm sound in a separate thread
def play_alarm():
    pygame.mixer.music.load('alarm_sound.mp3')  # Add the path to your alarm sound file here
    pygame.mixer.music.play()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for the face
        face_ROI_gray = gray[y:y+h, x:x+w]
        face_ROI_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_ROI_gray)

        if len(eyes) >= 2:  # We expect to detect 2 eyes for normal "open eyes"
            frame_counter = 0  # Reset the frame counter when eyes are detected
            ALARM_TRIGGERED = False  # Reset the alarm trigger flag
            for (ex, ey, ew, eh) in eyes[:2]:  # Only process the first two detected eyes
                # Draw rectangle around the eyes
                cv2.rectangle(face_ROI_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            # If no eyes are detected, increment the counter
            frame_counter += 1

        # Check if the eyes are closed for sufficient number of frames
        if frame_counter >= FRAMES_THRESHOLD:
            cv2.putText(frame, "SLEEP DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Play alarm if not already triggered
            if not ALARM_TRIGGERED:
                ALARM_TRIGGERED = True
                threading.Thread(target=play_alarm).start()

    # Display the frame
    cv2.imshow('Sleep Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
