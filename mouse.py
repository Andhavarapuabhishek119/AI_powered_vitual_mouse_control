import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time

# Initialize camera and MediaPipe
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# For smoothing nose movement
nose_points = deque(maxlen=5)

# State variables
input_mode = False
last_toggle_time = 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # ==== Draw Iris Points (Cursor Control) ====
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1 and input_mode:
                # Smoothing with nose instead (below)
                pass

        # ==== Eye Blink Detection for Click ====
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

        # ==== Nose-Based Cursor Movement ====
        nose = landmarks[1]  # Nose tip
        nx, ny = int(nose.x * frame_w), int(nose.y * frame_h)
        nose_points.append((nx, ny))

        if input_mode:
            avg_nx = int(np.mean([p[0] for p in nose_points]))
            avg_ny = int(np.mean([p[1] for p in nose_points]))
            screen_x = screen_w * (avg_nx / frame_w)
            screen_y = screen_h * (avg_ny / frame_h)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

        # ==== Mouth Open Detection to Toggle Mode ====
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        mouth_open = abs(top_lip.y - bottom_lip.y)

        if mouth_open > 0.05:
            if time.time() - last_toggle_time > 1:  # 1 second delay to avoid double toggle
                input_mode = not input_mode
                last_toggle_time = time.time()

        # ==== Head Tilt Scrolling ====
        left_forehead = landmarks[71]
        right_forehead = landmarks[301]
        tilt = left_forehead.y - right_forehead.y

        if input_mode:
            if tilt > 0.02:
                pyautogui.scroll(-30)  # Scroll down
            elif tilt < -0.02:
                pyautogui.scroll(30)   # Scroll up

        # ==== Status Display ====
        cv2.circle(frame, (nx, ny), 5, (255, 255, 0), -1)
        cv2.putText(frame, f'Mode: {"ON" if input_mode else "OFF"}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
