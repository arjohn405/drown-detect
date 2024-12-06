import os
import cv2
import dlib
from scipy.spatial import distance
import pygame
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

# Initialize pygame for the alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Replace with your alarm sound file

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Distance between upper lip and lower lip
    B = distance.euclidean(mouth[4], mouth[8])   # Distance between left corner and right corner of the mouth
    C = distance.euclidean(mouth[0], mouth[6])   # Distance between corners of the mouth
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CONSECUTIVE_FRAMES = 20
YAWN_WARNING_FRAMES = 30

# Counters
frame_count = 0
yawn_frame_count = 0

# Initialize decision tree classifier
dt_clf = DecisionTreeClassifier()

# Simulated training data for blinking classification (you can update it with real data)
X_train = np.array([[0.2], [0.3], [0.25], [0.4], [0.15]])  # EAR values
y_train = np.array([1, 0, 1, 0, 1])  # 1 = blink, 0 = no blink
dt_clf.fit(X_train, y_train)

# Load dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt",  # Path to gender model prototxt
    "gender_net.caffemodel"    # Path to pre-trained gender model weights
)
GENDER_LIST = ['Male', 'Female']

# Get indexes of left and right eye landmarks
LEFT_EYE_INDEXES = list(range(36, 42))
RIGHT_EYE_INDEXES = list(range(42, 48))
MOUTH_INDEXES = list(range(48, 68))

# Real-time variables for precision and recall calculation
y_true = []
y_pred = []

# Matplotlib setup for real-time graphs
plt.ion()  # Turn on interactive mode for live updates
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
precision_data = []
recall_data = []

# Function to update precision and recall graphs
def update_plot():
    ax1.clear()
    ax2.clear()
    ax1.plot(precision_data, label="Precision", color="blue")
    ax2.plot(recall_data, label="Recall", color="green")
    ax1.set_title("Precision over Time")
    ax2.set_title("Recall over Time")
    ax1.legend()
    ax2.legend()
    plt.draw()
    plt.pause(0.01)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Extract face ROI for gender detection
        face_roi = frame[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.42, 87.76, 114.89), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Display gender
        cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Extract landmarks for eyes and mouth
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_INDEXES]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_INDEXES]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH_INDEXES]

        # Draw landmark dots
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        # Calculate EAR and MAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(mouth)

        # Decision Tree for blinking classification
        prediction = dt_clf.predict([[ear]])
        y_true.append(1 if ear < EAR_THRESHOLD else 0)
        y_pred.append(prediction[0])

        # Drowsiness detection
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
        else:
            frame_count = 0
            pygame.mixer.music.stop()

        # Yawning detection
        if mar > MAR_THRESHOLD:
            yawn_frame_count += 1
            if yawn_frame_count >= YAWN_WARNING_FRAMES:
                cv2.putText(frame, "YAWNING DETECTED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
        else:
            yawn_frame_count = 0

    # Precision and recall calculation
    if len(y_true) > 1:
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        precision_data.append(precision)
        recall_data.append(recall)
        update_plot()

    # Display frame
    cv2.imshow("Drowsiness, Yawn, Gender Detection with Graphs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
