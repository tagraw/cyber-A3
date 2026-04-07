import cv2
import numpy as np

RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_BALL_AREA = 300
POSITION_THRESHOLD = 20
REFERENCE_AREA = 3000


def detect_red_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BALL_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(largest)
    return (x + w // 2, y + h // 2, x, y, w, h), mask


def determine_position(cx, cy, frame_w, frame_h, ball_area):
    dx = cx - frame_w // 2
    dy = cy - frame_h // 2
    abs_dx, abs_dy = abs(dx), abs(dy)

    # Depth from ball size
    if ball_area > REFERENCE_AREA * 1.3:
        depth_dir = "Front"
    elif ball_area < REFERENCE_AREA * 0.7:
        depth_dir = "Back"
    else:
        depth_dir = None

    # Both axes within threshold → centered (check depth only)
    if abs_dx < POSITION_THRESHOLD and abs_dy < POSITION_THRESHOLD:
        return depth_dir if depth_dir else "Centered"

    # Dominant axis wins; ties go horizontal
    if abs_dx >= abs_dy:
        if abs_dx > POSITION_THRESHOLD:
            return "Right" if dx > 0 else "Left"
    else:
        if abs_dy > POSITION_THRESHOLD:
            return "Down" if dy > 0 else "Up"

    return depth_dir if depth_dir else "Centered"


def draw_results(frame, cx, cy, x, y, w, h, position, ball_area):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    size = 10
    cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
    cv2.line(frame, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)

    cv2.putText(frame, position, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Area: {ball_area}", (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    fh, fw = frame.shape[:2]
    cv2.drawMarker(frame, (fw // 2, fh // 2), (255, 255, 0),
                   cv2.MARKER_CROSS, 20, 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Red ball tracker running. Press 'q' to quit.")
    print(f"REFERENCE_AREA = {REFERENCE_AREA}  (adjust if 'Close/Far' triggers wrong)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_h, frame_w = frame.shape[:2]
        result, mask = detect_red_ball(frame)

        if result is not None:
            cx, cy, x, y, w, h = result
            ball_area = w * h
            position = determine_position(cx, cy, frame_w, frame_h, ball_area)
            draw_results(frame, cx, cy, x, y, w, h, position, ball_area)
            print(f"Position: {position:20s}  Area: {ball_area}")
        else:
            cv2.putText(frame, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Red Ball Tracking", frame)
        cv2.imshow("Red Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()