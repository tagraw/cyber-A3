import argparse
import socket
import struct
import time
import numpy as np
import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

parser = argparse.ArgumentParser()
parser.add_argument("-n", default="192.168.4.1", metavar="ip")
parser.add_argument("-p", type=int, default=5000, metavar="port")
parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7", metavar="uri")
args = parser.parse_args()

RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_BALL_AREA = 300
POSITION_THRESHOLD = 20
REFERENCE_AREA = 3000
VELOCITY = 0.15
TAKEOFF_HEIGHT = 0.5


def detect_red_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
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

    if ball_area > REFERENCE_AREA * 1.3:
        depth_dir = "Front"
    elif ball_area < REFERENCE_AREA * 0.7:
        depth_dir = "Back"
    else:
        depth_dir = None

    abs_dx, abs_dy = abs(dx), abs(dy)

    if abs_dx < POSITION_THRESHOLD and abs_dy < POSITION_THRESHOLD:
        return depth_dir if depth_dir else "Centered"
    if abs_dx >= abs_dy and abs_dx > POSITION_THRESHOLD:
        return "Right" if dx > 0 else "Left"
    if abs_dy > abs_dx and abs_dy > POSITION_THRESHOLD:
        return "Down" if dy > 0 else "Up"

    return depth_dir if depth_dir else "Centered"


def move_drone(mc, position):
    if position == "Left":
        mc.start_left(VELOCITY)
    elif position == "Right":
        mc.start_right(VELOCITY)
    elif position == "Up":
        mc.start_up(VELOCITY)
    elif position == "Down":
        mc.start_down(VELOCITY)
    elif position == "Front":
        mc.start_back(VELOCITY)
    elif position == "Back":
        mc.start_forward(VELOCITY)
    else:
        mc.stop()


def draw_results(frame, cx, cy, x, y, w, h, position):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    size = 10
    cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
    cv2.line(frame, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)
    cv2.putText(frame, position, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    fh, fw = frame.shape[:2]
    cv2.drawMarker(frame, (fw // 2, fh // 2), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)


def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        data.extend(sock.recv(size - len(data)))
    return data


def get_frame(sock):
    packetInfoRaw = rx_bytes(sock, 4)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
    imgHeader = rx_bytes(sock, length - 2)
    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic != 0xBC:
        return None

    imgStream = bytearray()
    while len(imgStream) < size:
        packetInfoRaw = rx_bytes(sock, 4)
        [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
        imgStream.extend(rx_bytes(sock, length - 2))

    if format == 0:
        bayer_img = np.frombuffer(imgStream, dtype=np.uint8).reshape((244, 324))
        return cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
    else:
        nparr = np.frombuffer(imgStream, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


print(f"Connecting to AI deck at {args.n}:{args.p}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((args.n, args.p))
print("Socket connected")

cflib.crtp.init_drivers()
with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
    with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as mc:
        print("Crazyflie airborne. Tracking red ball. Press 'q' to land.")
        time.sleep(1)

        while True:
            frame = get_frame(client_socket)
            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]
            result, mask = detect_red_ball(frame)

            if result is not None:
                cx, cy, x, y, w, h = result
                position = determine_position(cx, cy, frame_w, frame_h, w * h)
                draw_results(frame, cx, cy, x, y, w, h, position)
                print(position)
                #move_drone(mc, position)
            else:
                mc.stop()
                cv2.putText(frame, "No ball detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Red Ball Tracking", frame)
            cv2.imshow("Red Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
client_socket.close()