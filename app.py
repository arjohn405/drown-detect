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
MAR_THRESHOLD = 0.6  # Threshold for yawning detection (adjust as needed)
CONSECUTIVE_FRAMES = 20
YAWN_WARNING_FRAMES = 30
BLINK_WARNING_FRAMES = 50

# Counters
frame_count = 0
blink_count = 0
yawn_frame_count = 0
frequent_blink_frame_count = 0

# Initialize decision tree classifier
dt_clf = DecisionTreeClassifier()

# Simulated training data for blinking classification (you can update it with real data)
X_train = np.array([[0.2], [0.3], [0.25], [0.4], [0.15]])  # EAR values
y_train = np.array([1, 0, 1, 0, 1])  # 1 = blink, 0 = no blink
dt_clf.fit(X_train, y_train)

# Load dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get indexes of left and right eye landmarks
LEFT_EYE_INDEXES = list(range(36, 42))
RIGHT_EYE_INDEXES = list(range(42, 48))
MOUTH_INDEXES = list(range(48, 68))

# Start video capture
cap = cv2.VideoCapture(0)

# Real-time variables for precision and recall calculation
y_true = []
y_pred = []

# CSV file for storing precision and recall scores
csv_file = "precision_recall_scores.csv"
# Write the header to the CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Precision", "Recall"])

# Matplotlib figure for plotting
plt.ion()  # Turn on interactive mode for real-time plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
precision_data = []
recall_data = []
time_data = []

# Function to update plot
def update_plot():
    ax1.clear()
    ax2.clear()
    ax1.plot(range(len(precision_data)), precision_data, label='Precision', color='blue')
    ax2.plot(range(len(recall_data)), recall_data, label='Recall', color='green')

    ax1.set_title("Precision over Time")
    ax2.set_title("Recall over Time")
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Time (frames)')
    ax2.set_xlabel('Time (frames)')
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('Recall')
    plt.draw()
    plt.pause(0.1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract landmarks for eyes and mouth
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_INDEXES]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_INDEXES]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH_INDEXES]

        # Calculate EAR and MAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(mouth)

        # Decision Tree for blinking classification
        prediction = dt_clf.predict([[ear]])

        # Append predicted and true values (simulated true values)
        y_true.append(1 if ear < EAR_THRESHOLD else 0)  # Assuming drowsiness based on EAR threshold
        y_pred.append(prediction[0])

        # Blink counting
        if prediction == 1:
            blink_count += 1

        # Check for drowsiness (based on EAR)
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
        else:
            frame_count = 0
            pygame.mixer.music.stop()

        # Frequent blinking warning
        if blink_count > 5 and frequent_blink_frame_count < BLINK_WARNING_FRAMES:
            cv2.putText(frame, "Frequent Blinking Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            frequent_blink_frame_count += 1
        else:
            frequent_blink_frame_count = 0

        # Yawning detection based on MAR
        if mar > MAR_THRESHOLD:
            yawn_frame_count += 1
            if yawn_frame_count >= YAWN_WARNING_FRAMES:
                cv2.putText(frame, "YAWNING DETECTED!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
        else:
            yawn_frame_count = 0

        # Draw eye and mouth landmarks
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Display frame
    cv2.imshow("Drowsiness and Yawn Detection", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate and save Precision and Recall periodically
    if len(y_true) > 10:  # Calculate after 10 frames of data
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, precision, recall])

        # Append precision and recall to data lists for plotting
        precision_data.append(precision)
        recall_data.append(recall)

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

        # Update the plot
        update_plot()

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
