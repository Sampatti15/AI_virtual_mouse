import cv2
import numpy as np
import hand_tracking_module as htm
import time
import autopy
import pyautogui
import os

# configuration
CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_MARGIN = 100
SMOOTH_FACTOR = 7
screenshot = "screenshots"


if not os.path.exists(screenshot):
    os.makedirs(screenshot)

prev_time = 0
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

camera = cv2.VideoCapture(0)
camera.set(3, CAM_WIDTH)
camera.set(4, CAM_HEIGHT)

hand_detector = htm.handDetector(maxHands=1)
SCREEN_WIDTH, SCREEN_HEIGHT = autopy.screen.size()

screenshot_count = 0
screenshot_delay = 0

while True:
    success, frame = camera.read()
    frame = hand_detector.findHands(frame)
    landmark_list, _ = hand_detector.findPosition(frame)

    if len(landmark_list) != 0:
        index_x, index_y = landmark_list[8][1:]
        middle_x, middle_y = landmark_list[12][1:]

        finger_status = hand_detector.fingersUp()

        cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                      (CAM_WIDTH - FRAME_MARGIN, CAM_HEIGHT - FRAME_MARGIN),
                      (255, 0, 255), 2)

        # move cursor (Only Index Finger)
        if finger_status[1] == 1 and finger_status[2] == 0:
            mapped_x = np.interp(index_x, (FRAME_MARGIN, CAM_WIDTH - FRAME_MARGIN),
                                 (0, SCREEN_WIDTH))
            mapped_y = np.interp(index_y, (FRAME_MARGIN, CAM_HEIGHT - FRAME_MARGIN),
                                 (0, SCREEN_HEIGHT))

            curr_x = prev_x + (mapped_x - prev_x) / SMOOTH_FACTOR
            curr_y = prev_y + (mapped_y - prev_y) / SMOOTH_FACTOR

            autopy.mouse.move(SCREEN_WIDTH - curr_x, curr_y)
            cv2.circle(frame, (index_x, index_y), 12, (255, 0, 255), cv2.FILLED)

            prev_x, prev_y = curr_x, curr_y

        # clickable mode  (Index + Middle) 
        if finger_status[1] == 1 and finger_status[2] == 1:
            distance, frame, info = hand_detector.findDistance(8, 12, frame)
            if distance < 40:
                cv2.circle(frame, (info[4], info[5]), 12, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                time.sleep(0.2)

        # screenshot   (Index + Middle + Little) 
        if finger_status[1] == 1 and finger_status[2] == 1 and finger_status[3] == 1:
            if screenshot_delay == 0:
                screenshot_count += 1
                filename = f"{screenshot}/screenshot_{screenshot_count}.png"
                pyautogui.screenshot(filename)
                screenshot_delay = 20
                cv2.putText(frame, "Screenshot Taken!", (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if screenshot_delay > 0:
        screenshot_delay -= 1

    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("AI Virtual Mouse", frame)
    cv2.waitKey(1)
