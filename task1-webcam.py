import cv2
import time

# Open default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

start = time.time()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    count += 1
    elapsed = time.time() - start
    fps = count / elapsed if elapsed > 0 else 0

    # -------------------------------
    # IMAGE PROCESSING
    # -------------------------------

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Overlay edges on original
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame, 0.8, edges_colored, 0.5, 0)

    # FPS display
    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # -------------------------------
    # DISPLAY WINDOWS
    # -------------------------------
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edges", edges)
    cv2.imshow("Overlay", overlay)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()