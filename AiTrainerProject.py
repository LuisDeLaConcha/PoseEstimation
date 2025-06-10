import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("AiTrainer/guyExercise.mp4")

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (2000, 3000))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right Arm
        angle_right = detector.findAngle(img, 12, 14, 16)
        per_right = np.interp(angle_right, (190, 70), (0, 100))  # Adjust the angle ranges if needed
        bar = np.interp(per_right, (0, 100), (1000, 400))  # Map percentage to bar position smoothly

        # Check for the curl count
        color = (255, 0, 255)  # Default color
        if per_right > 90:  # Upper limit threshold for counting
            color = (0, 255, 0)  # Change color to green
            if dir == 0:
                count += 0.5
                dir = 1
        if per_right < 10:   # Lower limit threshold for counting
            color = (0, 255, 0)  # Change color to green
            if dir == 1:
                count += 0.5
                dir = 0
        print(f"Count: {count}")

        # Draw Bar
        cv2.rectangle(img, (1700, 400), (1775, 1000), (0, 255, 0), 3)  # Static bar outline
        cv2.rectangle(img, (1700, int(bar)), (1775, 1000), color, cv2.FILLED)  # Moving bar based on percentage
        cv2.putText(img, f'{int(per_right)} %', (1600, 300), cv2.FONT_HERSHEY_PLAIN, 10,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (45, 670), cv2.FONT_HERSHEY_PLAIN, 10,
                    (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (180, 250), cv2.FONT_HERSHEY_PLAIN, 15,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
