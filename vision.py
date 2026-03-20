import cv2
import numpy as np
from picamera2 import Picamera2
import time

# ---------------- CAMERA SETUP ----------------
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
cam.start()
time.sleep(2)  # camera warm-up

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2

# ---------------- HSV RANGE (TUNE THIS) ----------------
# Example: YELLOW object
LOWER_HSV = np.array([0, 90, 100])
UPPER_HSV = np.array([95, 101, 108])

print("[INFO] Press 'q' to exit")

# ---------------- MAIN LOOP ----------------
while True:
    frame = cam.capture_array()

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # Clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 800:  # noise threshold
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

            error_x = cx - CENTER_X

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw frame center line
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 0, 0), 2)

            # Overlay diagnostics
            cv2.putText(
                frame,
                f"X Error: {error_x}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Vision Tracking Test", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cam.stop()
cv2.destroyAllWindows()
