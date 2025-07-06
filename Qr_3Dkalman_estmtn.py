import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Settings
TARGET_QR_DATA = "Vidhi's qr"
QR_REAL_SIZE = 2.0  # cm

# Camera parameters
FOCAL_LENGTH = 640.0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

# 3D points of QR code corners
OBJECT_POINTS = np.array([
    [-QR_REAL_SIZE/2, -QR_REAL_SIZE/2, 0],
    [QR_REAL_SIZE/2, -QR_REAL_SIZE/2, 0],
    [QR_REAL_SIZE/2, QR_REAL_SIZE/2, 0],
    [-QR_REAL_SIZE/2, QR_REAL_SIZE/2, 0]
], dtype=np.float32)

# Camera matrix
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, CENTER_X],
    [0, FOCAL_LENGTH, CENTER_Y],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(6, 3)
kalman.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]], np.float32)
kalman.transitionMatrix = np.array([
    [1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1],
    [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]
], np.float32)
kalman.processNoiseCov = 1e-5 * np.eye(6, dtype=np.float32)
kalman.measurementNoiseCov = 1e-3 * np.eye(3, dtype=np.float32)
kalman.statePost = np.zeros((6, 1), dtype=np.float32)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Detect QR codes
    decoded_objects = decode(frame)
    target_detected = False

    for obj in decoded_objects:
        qr_data = obj.data.decode("utf-8")
        if qr_data == TARGET_QR_DATA:
            target_detected = True
            qr_corners = np.array(obj.polygon, dtype=np.float32).reshape(-1, 2)
            qr_center = np.mean(qr_corners, axis=0)  # QR center in pixels

            if len(qr_corners) == 4:
                # Get raw pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    OBJECT_POINTS, qr_corners,
                    CAMERA_MATRIX, DIST_COEFFS
                )

                if success:
                    # Kalman update
                    measurement = np.array([
                        [tvec[0][0]],
                        [tvec[1][0]],
                        [tvec[2][0]]
                    ], dtype=np.float32)
                    prediction = kalman.predict()
                    kalman.correct(measurement)
                    smoothed_state = kalman.statePost
                    smoothed_tvec = smoothed_state[:3]
                    smoothed_distance = smoothed_tvec[2][0]

                    # Get rotation angles
                    rotation_mat, _ = cv2.Rodrigues(rvec)
                    angles_deg = np.degrees(cv2.RQDecomp3x3(rotation_mat)[0])
                    roll, pitch, yaw = angles_deg

                    # Calculate directional offsets (QR-center relative)
                    x_offset = qr_center[0] - CENTER_X
                    y_offset = CENTER_Y - qr_center[1]  # Flip Y-axis (up=positive)

                    # Determine direction instructions
                    x_dir = "RIGHT" if x_offset > 0 else "LEFT"
                    y_dir = "UP" if y_offset > 0 else "DOWN"
                    direction_text = f"Move {x_dir} {abs(x_offset):.0f}px, {y_dir} {abs(y_offset):.0f}px"

                    # Draw 3D axes at QR center
                    axis_points = np.float32([
                        [0,0,0], [1,0,0], [0,1,0], [0,0,-1]
                    ]) * QR_REAL_SIZE
                    img_points, _ = cv2.projectPoints(
                        axis_points, rvec, smoothed_tvec,
                        CAMERA_MATRIX, DIST_COEFFS
                    )
                    origin = tuple(map(int, img_points[0].ravel()))
                    cv2.line(frame, origin, tuple(map(int, img_points[1].ravel())), (0,0,255), 3)  # X
                    cv2.line(frame, origin, tuple(map(int, img_points[2].ravel())), (0,255,0), 3)  # Y
                    cv2.line(frame, origin, tuple(map(int, img_points[3].ravel())), (255,0,0), 3)  # Z

                    # Display information (QR-center relative)
                    cv2.putText(frame, f"Distance: {smoothed_distance:.1f} cm", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.putText(frame, direction_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)
                    cv2.putText(frame, f"Rotation: R:{roll:.1f}°, P:{pitch:.1f}°, Y:{yaw:.1f}°",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                    # Speed control
                    if smoothed_distance > 30:
                        cv2.putText(frame, "SPEED UP", (250, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    elif smoothed_distance < 10:
                        cv2.putText(frame, "SLOW DOWN", (250, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Draw QR boundary
            cv2.polylines(frame, [np.int32(obj.polygon)], True, (0,255,0), 2)
            
            # Draw crosshair at QR center
            cv2.drawMarker(frame, tuple(map(int, qr_center)), (0,255,255), cv2.MARKER_CROSS, 20, 2)

    if not target_detected:
        # Predict during occlusions
        prediction = kalman.predict()
        smoothed_tvec = prediction[:3]
        smoothed_distance = smoothed_tvec[2][0]
        
        cv2.putText(frame, "Target QR Not Found", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Last Distance: {smoothed_distance:.1f} cm", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Draw frame center crosshair
    cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255,0,255), cv2.MARKER_CROSS, 20, 2)
    
    cv2.imshow("QR Tracking with Navigation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()