import cv2
import mediapipe as mp
import time
import math


class handDetector:
    def __init__(self,
                 staticMode=False,
                 maxHands=2,
                 detectionConfidence=0.5,
                 trackingConfidence=0.5):

        self.staticMode = staticMode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.landmarkList = []

    # find hand
    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return image

    # to find position
    def findPosition(self, image, handNo=0, draw=True):
        xList, yList = [], []
        bbox = []
        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNo]

            h, w, _ = image.shape
            for idx, lm in enumerate(selectedHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.landmarkList.append([idx, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(image,
                              (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.landmarkList, bbox

    # fingers up
    def fingersUp(self):
        fingers = []

        if len(self.landmarkList) == 0:
            return fingers

        # Thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for i in range(1, 5):
            if self.landmarkList[self.tipIds[i]][2] < self.landmarkList[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    # distance
    def findDistance(self, p1, p2, image, draw=True, r=15, t=3):
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(image, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, image, [x1, y1, x2, y2, cx, cy]


# test module
def main():
    prevTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[8])  # index finger tip

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f'FPS: {int(fps)}',
                    (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
