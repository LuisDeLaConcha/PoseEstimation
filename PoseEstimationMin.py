import cv2
import mediapipe as mp
import time

# Correct path with "PoseVideos"
cap = cv2.VideoCapture('PoseVideos/1.mp4')

pTime = 0

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read video file")
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
