import cv2
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# ---------------- Load Models ----------------
def load_model(path):
    return YOLO(path)

vehicle_model = load_model(r'../Models/vehicle_detection.pt')
traffic_light_model = load_model(r'../Models/Traffic_Light_Detection.pt')

tracker = DeepSort(
    max_age=70,
    n_init=5,
    max_iou_distance=0.5,
    embedder="mobilenet"
)

violated_ids = set()
screenshot_dir = r".\Violation_ScreenShots"
os.makedirs(screenshot_dir, exist_ok=True)

# ---------------- Zebra Crossing ROI ----------------
def zebra_crossing_roi(video_path, display_width=1280, display_height=720):
    points = []
    drawing_done = False
    video = cv2.VideoCapture(video_path)
    ret, first_frame = video.read()
    if not ret:
        print("Cannot access video")
        video.release()
        cv2.destroyAllWindows()
        return None
    orig_height, orig_width = first_frame.shape[:2]
    scale_x = display_width / orig_width
    scale_y = display_height / orig_height

    def click_event(event, x, y, flags, param):
        nonlocal points, drawing_done
        if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
            orig_x = int(x / scale_x)
            orig_y = int(y / scale_y)
            points.append((orig_x, orig_y))
            if len(points) == 4:
                drawing_done = True

    cv2.namedWindow("Select Zebra Crossing (Polygon)")
    cv2.setMouseCallback("Select Zebra Crossing (Polygon)", click_event)

    while True:
        temp_frame = first_frame.copy()
        for pt in points:
            cv2.circle(temp_frame, pt, 6, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        if len(points) == 4:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(temp_frame, "Press 'c' to continue or 'a' to reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        display_frame = cv2.resize(temp_frame, (display_width, display_height))
        cv2.imshow("Select Zebra Crossing (Polygon)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return None
        elif drawing_done and key == ord('c'):
            break
        elif key == ord('a'):
            points = []
            drawing_done = False
    cv2.destroyWindow("Select Zebra Crossing (Polygon)")
    video.release()
    return np.array(points)

# ---------------- Polygon Violation Check ----------------
def is_vehicle_crossing(box, polygon_points):
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return cv2.pointPolygonTest(polygon_points, (x_center, y_center), False) >= 0

# ---------------- Video Processing ----------------
def process_video(video_path, polygon_points, display_width=1280, display_height=720):
    global violated_ids
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Draw polygon ROI
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in polygon_points:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)

        # Traffic Light Detection
        traffic_results = traffic_light_model.predict(frame, conf=0.5, verbose=False)
        red_light = False
        for r in traffic_results:
            for box_obj in r.boxes:
                cls_id = int(box_obj.cls[0])
                label = r.names[cls_id]
                box = box_obj.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if label in ['Red', 'RedRight'] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if label in ['Red', 'RedRight']:
                    red_light = True

        # Vehicle Detection + Tracking
        if red_light:
            vehicle_results = vehicle_model.predict(frame, conf=0.15, verbose=False)[0]
            detections = []
            for box_obj in vehicle_results.boxes:
                box = box_obj.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                conf = float(box_obj.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "vehicle"))
            tracks = tracker.update_tracks(detections, frame=frame)

            violators = []

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Check violation
                if is_vehicle_crossing([x1, y1, x2, y2], polygon_points):
                    if track_id not in violated_ids:
                        violated_ids.add(track_id)
                        # Save screenshot
                        ss = frame.copy()
                        cv2.rectangle(ss, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(ss, f"ID {track_id} - VIOLATED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        screenshot_path = os.path.join(screenshot_dir, f"violation_{track_id}.jpg")
                        cv2.imwrite(screenshot_path, ss)
                        print(f"Screenshot saved: {screenshot_path}")

                    violators.append(f"ID {track_id}")

                # Draw bounding box + ID
                color = (0, 0, 255) if track_id in violated_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display top-left violators with background
            if violators:
                text = "Violators: " + ", ".join(violators)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (5,5), (10+w, 10+h), (0,0,0), -1)  # background
                cv2.putText(frame, text, (10, 10+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("Video Processing", display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# ---------------- Run Pipeline ----------------
if __name__=='__main__':
    video_path = r"C:\Users\UqashaZahid\Downloads\test_3.mp4"
    polygon_points = zebra_crossing_roi(video_path)
    if polygon_points is not None:
        process_video(video_path, polygon_points)