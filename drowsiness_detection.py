# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial import distance as dist
# import pyttsx3

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Define a function to calculate the eye aspect ratio (EAR)
# def eye_aspect_ratio(eye_landmarks):
#     A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
#     B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
#     C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Define the eye aspect ratio threshold and consecutive frame count threshold
# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 30

# # Initialize the frame counters and the total number of blinks
# COUNTER = 0
# ALARM_ON = False

# # Start video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark
#             height, width, _ = frame.shape

#             left_eye_landmarks = [(int(landmarks[362].x * width), int(landmarks[362].y * height)),
#                                   (int(landmarks[385].x * width), int(landmarks[385].y * height)),
#                                   (int(landmarks[387].x * width), int(landmarks[387].y * height)),
#                                   (int(landmarks[263].x * width), int(landmarks[263].y * height)),
#                                   (int(landmarks[373].x * width), int(landmarks[373].y * height)),
#                                   (int(landmarks[380].x * width), int(landmarks[380].y * height))]

#             right_eye_landmarks = [(int(landmarks[33].x * width), int(landmarks[33].y * height)),
#                                    (int(landmarks[160].x * width), int(landmarks[160].y * height)),
#                                    (int(landmarks[158].x * width), int(landmarks[158].y * height)),
#                                    (int(landmarks[133].x * width), int(landmarks[133].y * height)),
#                                    (int(landmarks[153].x * width), int(landmarks[153].y * height)),
#                                    (int(landmarks[144].x * width), int(landmarks[144].y * height))]

#             leftEAR = eye_aspect_ratio(left_eye_landmarks)
#             rightEAR = eye_aspect_ratio(right_eye_landmarks)

#             ear = (leftEAR + rightEAR) / 2.0

#             for point in left_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)
#             for point in right_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)

#             if ear < EYE_AR_THRESH:
#                 COUNTER += 1
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                     if not ALARM_ON:
#                         ALARM_ON = True
#                         engine.say("Drowsiness detected! Please take a break.")
#                         engine.runAndWait()
#                         print("Drowsiness detected!")
#                     cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 COUNTER = 0
#                 ALARM_ON = False

#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial import distance as dist
# import pyttsx3

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Define a function to calculate the eye aspect ratio (EAR)
# def eye_aspect_ratio(eye_landmarks):
#     A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
#     B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
#     C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Define the eye aspect ratio threshold and consecutive frame count threshold
# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 48

# # Initialize the frame counters and the total number of blinks
# COUNTER = 0
# ALARM_ON = False

# def process_frame(frame):
#     global COUNTER, ALARM_ON
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark
#             height, width, _ = frame.shape

#             # Extract eye landmarks
#             left_eye_landmarks = [(int(landmarks[362].x * width), int(landmarks[362].y * height)),
#                                   (int(landmarks[385].x * width), int(landmarks[385].y * height)),
#                                   (int(landmarks[387].x * width), int(landmarks[387].y * height)),
#                                   (int(landmarks[263].x * width), int(landmarks[263].y * height)),
#                                   (int(landmarks[373].x * width), int(landmarks[373].y * height)),
#                                   (int(landmarks[380].x * width), int(landmarks[380].y * height))]

#             right_eye_landmarks = [(int(landmarks[33].x * width), int(landmarks[33].y * height)),
#                                    (int(landmarks[160].x * width), int(landmarks[160].y * height)),
#                                    (int(landmarks[158].x * width), int(landmarks[158].y * height)),
#                                    (int(landmarks[133].x * width), int(landmarks[133].y * height)),
#                                    (int(landmarks[153].x * width), int(landmarks[153].y * height)),
#                                    (int(landmarks[144].x * width), int(landmarks[144].y * height))]

#             leftEAR = eye_aspect_ratio(left_eye_landmarks)
#             rightEAR = eye_aspect_ratio(right_eye_landmarks)

#             ear = (leftEAR + rightEAR) / 2.0

#             # Draw eye landmarks
#             for point in left_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)
#             for point in right_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)

#             # Check if EAR is below threshold
#             if ear < EYE_AR_THRESH:
#                 COUNTER += 1
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                     if not ALARM_ON:
#                         ALARM_ON = True
#                         engine.say("Drowsiness detected! Please take a break.")
#                         engine.runAndWait()
#                         print("Drowsiness detected!")
#                     cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 COUNTER = 0
#                 ALARM_ON = False

#     return frame

# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial import distance as dist
# import pyttsx3
# import threading

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Define a function to calculate the eye aspect ratio (EAR)
# def eye_aspect_ratio(eye_landmarks):
#     A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
#     B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
#     C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Define the eye aspect ratio threshold and consecutive frame count threshold
# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 48

# # Initialize the frame counters and the total number of blinks
# COUNTER = 0
# ALARM_ON = False

# def process_frame(frame):
#     global COUNTER, ALARM_ON

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark
#             height, width, _ = frame.shape

#             left_eye_landmarks = [(int(landmarks[362].x * width), int(landmarks[362].y * height)),
#                                   (int(landmarks[385].x * width), int(landmarks[385].y * height)),
#                                   (int(landmarks[387].x * width), int(landmarks[387].y * height)),
#                                   (int(landmarks[263].x * width), int(landmarks[263].y * height)),
#                                   (int(landmarks[373].x * width), int(landmarks[373].y * height)),
#                                   (int(landmarks[380].x * width), int(landmarks[380].y * height))]

#             right_eye_landmarks = [(int(landmarks[33].x * width), int(landmarks[33].y * height)),
#                                    (int(landmarks[160].x * width), int(landmarks[160].y * height)),
#                                    (int(landmarks[158].x * width), int(landmarks[158].y * height)),
#                                    (int(landmarks[133].x * width), int(landmarks[133].y * height)),
#                                    (int(landmarks[153].x * width), int(landmarks[153].y * height)),
#                                    (int(landmarks[144].x * width), int(landmarks[144].y * height))]

#             leftEAR = eye_aspect_ratio(left_eye_landmarks)
#             rightEAR = eye_aspect_ratio(right_eye_landmarks)

#             ear = (leftEAR + rightEAR) / 2.0

#             # Draw eye landmarks
#             for point in left_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)
#             for point in right_eye_landmarks:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)

#             if ear < EYE_AR_THRESH:
#                 COUNTER += 1
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                     if not ALARM_ON:
#                         ALARM_ON = True
#                         # Use a separate thread to avoid blocking the frame processing
#                         def speak():
#                             engine.say("Drowsiness detected! Please take a break.")
#                             engine.runAndWait()

#                         threading.Thread(target=speak).start()
#                         print("Drowsiness detected!")
#                     cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 COUNTER = 0
#                 if ALARM_ON:
#                     ALARM_ON = False
#                     print("Alarm reset.")

#     return frame

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pyttsx3
import threading

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
ALARM_ON = False

# Define a function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define the eye aspect ratio threshold and consecutive frame count threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# Initialize the frame counters and the total number of blinks
COUNTER = 0

def speak():
    engine.say("Drowsiness detected! Please take a break.")
    engine.runAndWait()

def process_frame(frame):
    global COUNTER, ALARM_ON

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            height, width, _ = frame.shape

            left_eye_landmarks = [(int(landmarks[362].x * width), int(landmarks[362].y * height)),
                                  (int(landmarks[385].x * width), int(landmarks[385].y * height)),
                                  (int(landmarks[387].x * width), int(landmarks[387].y * height)),
                                  (int(landmarks[263].x * width), int(landmarks[263].y * height)),
                                  (int(landmarks[373].x * width), int(landmarks[373].y * height)),
                                  (int(landmarks[380].x * width), int(landmarks[380].y * height))]

            right_eye_landmarks = [(int(landmarks[33].x * width), int(landmarks[33].y * height)),
                                   (int(landmarks[160].x * width), int(landmarks[160].y * height)),
                                   (int(landmarks[158].x * width), int(landmarks[158].y * height)),
                                   (int(landmarks[133].x * width), int(landmarks[133].y * height)),
                                   (int(landmarks[153].x * width), int(landmarks[153].y * height)),
                                   (int(landmarks[144].x * width), int(landmarks[144].y * height))]

            leftEAR = eye_aspect_ratio(left_eye_landmarks)
            rightEAR = eye_aspect_ratio(right_eye_landmarks)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye landmarks
            for point in left_eye_landmarks:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye_landmarks:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # Start the alert in a separate thread
                        threading.Thread(target=speak).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False

    return frame
