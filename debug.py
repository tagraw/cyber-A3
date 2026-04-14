import argparse
import socket
import struct
import time
import numpy as np
import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.high_level_commander import HighLevelCommander

parser = argparse.ArgumentParser()
parser.add_argument("-n", default="192.168.4.1", metavar="ip")
parser.add_argument("-p", type=int, default=5000, metavar="port")
parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E711", metavar="uri")
args = parser.parse_args()

# --- HSV ranges for red ball (color camera) ---
RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

# --- Tuning ---
MIN_BALL_AREA      = 300    # ignore blobs smaller than this
POSITION_THRESHOLD = 20     # px from center before moving
DEPTH_THRESHOLD    = 0.20   # ±20% diameter change triggers Front/Back
BINARY_THRESHOLD   = 60     # grayscale cutoff for mono detection

TAKEOFF_HEIGHT = 0.5
STEP           = 0.2    # metres per go_to step
MOVE_DURATION  = 1.0    # seconds for each go_to (must finish before next cmd)
CMD_INTERVAL   = 1.5    # seconds between commands (MOVE_DURATION + settling margin)

reference_diameter = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_ball_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return _largest_contour(mask), mask


def detect_ball_mono(frame):
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return _largest_contour(mask), mask


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BALL_AREA:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    diameter = (w + h) / 2
    return (x + w // 2, y + h // 2, x, y, w, h, diameter)


# ---------------------------------------------------------------------------
# Position logic
# ---------------------------------------------------------------------------

def determine_position(cx, cy, frame_w, frame_h, diameter):
    dx = cx - frame_w // 2
    dy = cy - frame_h // 2
    abs_dx, abs_dy = abs(dx), abs(dy)

    depth_dir = None
    if reference_diameter is not None:
        ratio = diameter / reference_diameter
        if ratio > 1 + DEPTH_THRESHOLD:
            depth_dir = "Front"
        elif ratio < 1 - DEPTH_THRESHOLD:
            depth_dir = "Back"

    if abs_dx < POSITION_THRESHOLD and abs_dy < POSITION_THRESHOLD:
        return depth_dir if depth_dir else "Centered"

    if abs_dx >= abs_dy:
        return "Right" if dx > 0 else "Left"
    else:
        # dy > 0 → ball below image centre → drone must go Down
        # dy < 0 → ball above image centre → drone must go Up
        return "Down" if dy > 0 else "Up"


# ---------------------------------------------------------------------------
# Drone control — HighLevelCommander with tracked absolute position
# ---------------------------------------------------------------------------

def move_drone(hlc, pos, position):
    """
    Mutates pos dict and issues a go_to.
    Body frame (matches working command code):
      x → forward / back
      y → left / right
      z → up / down
    """
    if position == "Left":
        pos["y"] += STEP
    elif position == "Right":
        pos["y"] -= STEP
    elif position == "Up":
        pos["z"] += STEP
    elif position == "Down":
        pos["z"] -= STEP
    elif position == "Front":      # ball too close → back up
        pos["x"] -= STEP
    elif position == "Back":       # ball too far  → move forward
        pos["x"] += STEP
    else:
        return  # Centered — hold position, no go_to needed

    hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], MOVE_DURATION)
    time.sleep(MOVE_DURATION + 0.5)   # wait for move to complete before next frame


# ---------------------------------------------------------------------------
# Manual commands (terminal input, mirrors main.py)
# ---------------------------------------------------------------------------

def log_stab_callback(timestamp, data, logconf):
    print(
        f"[{timestamp}] "
        f"Roll={data['stabilizer.roll']:.3f}, "
        f"Pitch={data['stabilizer.pitch']:.3f}, "
        f"Yaw={data['stabilizer.yaw']:.3f}"
    )


def handle_commands(cmd, scf, lg_stab, state):
    """
    Manual terminal commands (same as main.py):
      i          — log stabilizer for 2 s
      s          — take off to TAKEOFF_HEIGHT
      u <m>      — move up by <m> metres
      d <m>      — move down by <m> metres
      f <m>      — move forward by <m> metres
      b <m>      — move back by <m> metres
      l <deg>    — yaw left by <deg> degrees
      r <deg>    — yaw right by <deg> degrees
      n          — land and stop
      c          — enter camera-tracking mode (OpenCV window)
    Returns False to quit the command loop, True to continue.
    """
    parts = cmd.strip().split()
    if not parts:
        return True

    c = parts[0]

    # i — log stabiliser
    if c == 'i':
        scf.cf.log.add_config(lg_stab)
        lg_stab.data_received_cb.add_callback(log_stab_callback)
        lg_stab.start()
        time.sleep(2)
        lg_stab.stop()
        return True

    # s — take off
    if c == 's':
        if state["hlc"] is None:
            state["hlc"] = HighLevelCommander(scf.cf)
            state["hlc"].takeoff(TAKEOFF_HEIGHT, 2.0)
            time.sleep(2.5)
            state["pos"] = {"x": 0.0, "y": 0.0, "z": TAKEOFF_HEIGHT, "yaw": 0.0}
            print("Airborne.")
        else:
            print("Already airborne — use 'n' to land first.")
        return True

    # n — land
    if c == 'n':
        if state["hlc"] is None:
            print("Not airborne.")
            return True
        state["hlc"].land(0.0, 2.0)
        time.sleep(2.5)
        state["hlc"].stop()
        state["hlc"] = None
        state["pos"] = None
        print("Landed.")
        return True

    # Commands that require being airborne
    if state["hlc"] is None:
        print("Must take off first — use 's'.")
        return True

    hlc  = state["hlc"]
    pos  = state["pos"]

    # u — up
    if c == 'u':
        pos["z"] += float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # d — down
    if c == 'd':
        pos["z"] -= float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # f — forward
    if c == 'f':
        pos["x"] += float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # b — back
    if c == 'b':
        pos["x"] -= float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # l — yaw left
    if c == 'l':
        pos["yaw"] += float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # r — yaw right
    if c == 'r':
        pos["yaw"] -= float(parts[1])
        hlc.go_to(pos["x"], pos["y"], pos["z"], pos["yaw"], 1.0)
        time.sleep(1.5)
        return True

    # c — enter camera-tracking mode
    if c == 'c':
        run_camera_tracking(hlc, pos, state)
        return True

    print(f"Unknown command: '{c}'")
    return True


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_results(frame, cx, cy, x, y, w, h, diameter, position):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    size = 10
    cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
    cv2.line(frame, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)
    cv2.putText(frame, position, (x, max(y - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    diam_str = f"d:{diameter:.0f}"
    if reference_diameter is not None:
        diam_str += f" ref:{reference_diameter:.0f}"
    cv2.putText(frame, diam_str, (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    fh, fw = frame.shape[:2]
    cv2.drawMarker(frame, (fw // 2, fh // 2), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)

    if reference_diameter is None:
        cv2.putText(frame, "SPACE: calibrate depth", (5, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


def draw_hud(frame, tracking_active):
    """Overlay current mode and key hints on the camera feed."""
    fh, fw = frame.shape[:2]
    mode_str  = "MODE: TRACKING" if tracking_active else "MODE: OBSERVE  (press C to track)"
    mode_color = (0, 255, 0) if tracking_active else (0, 200, 255)
    cv2.putText(frame, mode_str, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    cv2.putText(frame, "C: toggle tracking  SPACE: calibrate  Q: exit camera", (5, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        data.extend(sock.recv(size - len(data)))
    return data


def get_frame(sock):
    packetInfoRaw = rx_bytes(sock, 4)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
    imgHeader = rx_bytes(sock, length - 2)
    [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic != 0xBC:
        return None, None

    imgStream = bytearray()
    while len(imgStream) < size:
        packetInfoRaw = rx_bytes(sock, 4)
        [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
        imgStream.extend(rx_bytes(sock, length - 2))

    if fmt == 0:
        # Raw Bayer → color
        bayer_img = np.frombuffer(imgStream, dtype=np.uint8).reshape((244, 324))
        return cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR), "color"
    else:
        arr = np.frombuffer(imgStream, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        is_mono = (img is not None) and (len(img.shape) == 2 or img.shape[2] == 1)
        return img, ("mono" if is_mono else "color")


# ---------------------------------------------------------------------------
# Camera-tracking loop  (entered via 'c' command)
# ---------------------------------------------------------------------------

def run_camera_tracking(hlc, pos, state):
    """
    Opens the OpenCV window and streams frames.
    Pressing 'c' inside the window toggles autonomous tracking on/off.
    Pressing 'q' closes the window and returns to the terminal command loop.
    """
    global reference_diameter

    tracking_active = False   # start in observe-only mode so you can look first
    last_cmd_time   = 0
    last_position   = None
    result          = None
    mask            = None

    print("\n--- Camera mode ---")
    print("  C      toggle autonomous tracking on/off")
    print("  SPACE  calibrate Front/Back reference distance")
    print("  Q      exit camera mode (return to terminal commands)")

    while True:
        frame, cam_mode = get_frame(client_socket)
        if frame is None:
            continue

        # Ensure BGR for drawing
        if len(frame.shape) == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        frame_h, frame_w = display.shape[:2]

        if cam_mode == "color":
            result, mask = detect_ball_color(frame)
        else:
            result, mask = detect_ball_mono(frame)

        current_time = time.time()

        if result is not None:
            cx, cy, x, y, w, h, diameter = result
            position = determine_position(cx, cy, frame_w, frame_h, diameter)
            draw_results(display, cx, cy, x, y, w, h, diameter, position)
            print(position)

            if tracking_active:
                if (current_time - last_cmd_time >= CMD_INTERVAL and
                        position != last_position):
                    move_drone(hlc, pos, position)
                    last_cmd_time = time.time()
                    last_position = position
        else:
            cv2.putText(display, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            last_position = None

        draw_hud(display, tracking_active)
        cv2.imshow("Ball Tracking", display)
        if mask is not None:
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting camera mode. Returning to terminal commands.")
            cv2.destroyAllWindows()
            break

        elif key == ord('c'):
            tracking_active = not tracking_active
            status = "ENABLED" if tracking_active else "DISABLED"
            print(f"Autonomous tracking {status}.")

        elif key == ord(' '):
            if result is not None:
                reference_diameter = result[6]
                print(f"Calibrated reference diameter: {reference_diameter:.1f} px")
            else:
                print("No ball in frame — move ball into view before calibrating.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f"Connecting to AI deck at {args.n}:{args.p}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((args.n, args.p))
print("Socket connected.")

cflib.crtp.init_drivers()

with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache='./cache')) as scf:

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
    lg_stab.add_variable('stabilizer.roll',  'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw',   'float')

    state = {"hlc": None, "pos": None}

    print("\nManual commands:")
    print("  i        — log stabiliser (2 s)")
    print("  s        — take off")
    print("  u/d <m>  — up / down")
    print("  f/b <m>  — forward / back")
    print("  l/r <°>  — yaw left / right")
    print("  c        — open camera & tracking mode")
    print("  n        — land")

    try:
        while True:
            cmd = input("> ")
            cont = handle_commands(cmd, scf, lg_stab, state)
            if cont is False:
                break
    finally:
        # Safety net: always land if still airborne when the script exits
        if state["hlc"] is not None:
            print("Safety landing...")
            state["hlc"].land(0.0, 2.0)
            time.sleep(2.5)
            state["hlc"].stop()

cv2.destroyAllWindows()
client_socket.close()