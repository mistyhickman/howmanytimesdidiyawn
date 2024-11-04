import cv2
import dlib
import numpy as np
import pygame
import time
from scipy.spatial import distance as dist

# Initialize Pygame for sound playback
pygame.mixer.init()
sound = pygame.mixer.Sound("./PopSound.wav")

# Load Dlib's pre-trained face detector and facial landmarks model (68 landmarks)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables
yawn_count = 0
yawn_threshold = 0.6  # Lip distance threshold for yawning
consecutive_yawn_frames = 0  # Frame count for consecutive frames indicating a yawn
required_yawn_frames = 15  # Frames needed to confirm a yawn
cooldown_period = 2  # Minimum time between yawns, in seconds
last_yawn_time = 0  # Track time between yawns
yawn_active = False  # Track if a yawn is currently active
close_mouth_frames = 0  # Frames to confirm mouth is closed

# Define function to calculate mouth aspect ratio (MAR)
def calculate_mouth_aspect_ratio(mouth_points):
    A = dist.euclidean(mouth_points[2], mouth_points[10])  # Vertical distance
    B = dist.euclidean(mouth_points[4], mouth_points[8])   # Vertical distance
    C = dist.euclidean(mouth_points[0], mouth_points[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Draw rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        try:
            # Extract mouth coordinates using Dlib's facial landmarks (points 48-67)
            mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            # Draw contours around the mouth for visualization
            cv2.drawContours(frame, [mouth_points], -1, (255, 0, 0), 1)

            # Calculate the Mouth Aspect Ratio (MAR)
            mar = calculate_mouth_aspect_ratio(mouth_points)

            # Detect yawn start based on MAR threshold
            if mar > yawn_threshold:
                consecutive_yawn_frames += 1
                close_mouth_frames = 0  # Reset the closed-mouth counter

                # Count yawn if it meets frame requirement and cooldown period
                if consecutive_yawn_frames >= required_yawn_frames and not yawn_active and (time.time() - last_yawn_time) > cooldown_period:
                    yawn_count += 1
                    last_yawn_time = time.time()
                    sound.play()
                    print("Yawn detected!")
                    yawn_active = True  # Mark yawn as active to avoid re-counting
            else:
                # Increment closed-mouth frames to confirm the end of a yawn
                close_mouth_frames += 1

                # Reset if mouth has been closed long enough
                if close_mouth_frames >= 5:  # Customize this as needed
                    yawn_active = False
                    consecutive_yawn_frames = 0

        except IndexError:
            # Skip processing if 68 landmark points are not detected
            print("Could not detect all facial landmarks. Skipping frame.")

    # Display yawn count on the screen
    cv2.putText(frame, f'Yawn Count: {yawn_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Show the frame with face and mouth rectangles
    cv2.imshow('Yawn Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Output total yawn count to the terminal
print(f"Total Yawns Recorded: {yawn_count}")
