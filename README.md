# howmanytimesdidiyawn

Yawn Detection Program
This project is a real-time yawn detection system built with Python. It uses a webcam to detect yawns based on mouth aspect ratio (MAR) calculations from facial landmarks. Just a bit of fun that I had when I couldn't sleep at 4am one day. Enjoy!

Features
Real-time yawn detection using a webcam feed.
Alerts with a sound effect when a yawn is detected.
Displays yawn count on the screen.

Prerequisites
Python 3.x
shape_predictor_68_face_landmarks.dat (included in the repository)
PopSound.wav (included in the repository)

Installation
Clone the repository:

git clone https://github.com/mistyhickman/howmanytimesdidiyawn

Install dependencies: If there's a .whl file for dependencies in the repository root, you can install it directly:

pip install <your-whl-file-name>.whl

Alternatively, install the required packages individually:

pip install opencv-python dlib pygame scipy numpy


Usage
Run the program:


python dlibmain.py
Instructions:

The program will access your default webcam.
A window will open showing a live feed with a rectangle around the detected face and contours around the mouth.
Each time a yawn is detected, a pop sound will play, and the yawn count will be updated on the screen.
Press q to quit the program.

Adjust Parameters:

yawn_threshold: Adjusts the MAR threshold for detecting a yawn.
required_yawn_frames: Sets the minimum number of consecutive frames to confirm a yawn.
cooldown_period: Minimum time in seconds between detected yawns to avoid multiple counts from the same yawn.

Project Structure
dlibmain.py: Main program file.
shape_predictor_68_face_landmarks.dat: Pre-trained facial landmarks model.
PopSound.wav: Sound effect played when a yawn is detected.