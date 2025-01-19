import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

class KalmanFilter:
    def __init__(self):
        self.state = None  # Placeholder for Kalman Filter state

    def predict(self):
        pass

    def update(self, detection):
        pass

class Tracker:
    def __init__(self, id, detection, class_id):
        self.id = id
        self.class_id = class_id
        self.kf = KalmanFilter()
        self.kf.update(detection)
        self.detection = detection
        self.lost_count = 0
        self.static_count = 0
        self.history_top_left = []
        self.history_bottom_right = []
        self.crossed_red_line = False  # Flag to indicate if the tracker has already been counted

class DeepSort:
    def __init__(self):
        self.trackers = []
        self.next_id = 0
        self.all_paths_top_left = []
        self.all_paths_bottom_right = []
        self.vehicle_count = 0
        self.path_detected = False
        self.refinement_needed = False
        self.vehicle_classes_count = [0] * 7  # Initialize counts for 7 classes
        self.red_line = None

    def update(self, detections, frame_count, classes):
        for tracker in self.trackers:
            tracker.kf.predict()

        assigned_detections = [False] * len(detections)
        cost_matrix = np.zeros((len(self.trackers), len(detections)))
        for i, tracker in enumerate(self.trackers):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(tracker.detection[:2] - detection[:2])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 50:
                tracker = self.trackers[i]
                tracker.kf.update(detections[j])
                tracker.detection = detections[j]
                tracker.lost_count = 0
                tracker.static_count = 0
                assigned_detections[j] = True
                
                x1, y1, w, h = detections[j]
                x2, y2 = x1 + w, y1 + h
                pt_top_left = (int(x1), int(y1))
                pt_bottom_right = (int(x2), int(y2))
                
                if frame_count % 30 == 0:
                    tracker.history_top_left.append(pt_top_left)
                    tracker.history_bottom_right.append(pt_bottom_right)
                
                if frame_count % 180 == 0:
                    self.all_paths_top_left.extend(tracker.history_top_left)
                    self.all_paths_bottom_right.extend(tracker.history_bottom_right)
                    tracker.history_top_left = []
                    tracker.history_bottom_right = []

        new_trackers = []
        for i, assigned in enumerate(assigned_detections):
            if not assigned:
                new_trackers.append(Tracker(self.next_id, detections[i], classes[i]))
                self.next_id += 1
                self.vehicle_count += 1
        self.trackers = [tracker for tracker in self.trackers if tracker.lost_count < 5 and tracker.static_count < 10]
        self.trackers.extend(new_trackers)

        for tracker in self.trackers:
            if tracker.static_count < 10:
                tracker.static_count += 1
            else:
                tracker.lost_count += 1

    def draw_paths(self, frame):
        for tracker in self.trackers:
            if len(tracker.history_top_left) > 1:
                for i in range(1, len(tracker.history_top_left)):
                    pt1, pt2 = tracker.history_top_left[i - 1], tracker.history_top_left[i]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                for i in range(1, len(tracker.history_bottom_right)):
                    pt1, pt2 = tracker.history_bottom_right[i - 1], tracker.history_bottom_right[i]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        if self.vehicle_count >= 10 and self.all_paths_top_left and self.all_paths_bottom_right:
            avg_top_left = np.mean(self.all_paths_top_left, axis=0).astype(int)
            avg_bottom_right = np.mean(self.all_paths_bottom_right, axis=0).astype(int)

            cv2.line(frame, tuple(avg_top_left), tuple(avg_bottom_right), (255, 0, 0), 2)
            cv2.line(frame, tuple(avg_bottom_right), tuple(avg_top_left), (255, 0, 0), 2)

            midpoint = ((avg_top_left[0] + avg_bottom_right[0]) // 2, (avg_top_left[1] + avg_bottom_right[1]) // 2)
            perp_slope = -1 / ((avg_bottom_right[1] - avg_top_left[1]) / (avg_bottom_right[0] - avg_top_left[0]))
            perp_length = 500
            dx = int(perp_length / (1 + perp_slope ** 2) ** 0.5)
            dy = int(perp_slope * dx)
            pt1 = (midpoint[0] - dx, midpoint[1] - dy)
            pt2 = (midpoint[0] + dx, midpoint[1] + dy)
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            self.red_line = (pt1, pt2)

            print(f"Red Line Points: {self.red_line}")  # Debugging line

            if self.refinement_needed:
                self.all_paths_top_left = []
                self.all_paths_bottom_right = []

    def count_vehicles_passing_red_line(self):
        if self.red_line:
            for tracker in self.trackers:
                x, y, w, h = tracker.detection
                midpoint = (int(x + w / 2), int(y + h / 2))

                # Debugging: Print vehicle midpoint and red line
                print(f"Vehicle ID: {tracker.id} Midpoint: {midpoint}")
                print(f"Red Line: {self.red_line}")

                if self.line_intersects_point(self.red_line[0], self.red_line[1], midpoint) and not tracker.crossed_red_line:
                    print(f"Vehicle ID: {tracker.id} with Class: {tracker.class_id} crossed the red line")  # Debugging line
                    self.vehicle_classes_count[tracker.class_id] += 1
                    tracker.crossed_red_line = True

    def line_intersects_point(self, pt1, pt2, point, threshold=5):
        """
        Check if a point is close to the line segment defined by pt1 and pt2.
        """
        def distance_point_line(px, py, x1, y1, x2, y2):
            """Calculate the minimum distance from point (px, py) to the line segment (x1, y1) - (x2, y2)."""
            line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if line_len_sq == 0:
                return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
            projx = x1 + t * (x2 - x1)
            projy = y1 + t * (y2 - y1)
            return np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
        
        return distance_point_line(point[0], point[1], pt1[0], pt1[1], pt2[0], pt2[1]) < threshold

    def print_vehicle_count(self, frame):
        class_names = ["Bicycle", "Bus", "Cars", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"]
        for i, count in enumerate(self.vehicle_classes_count):
            cv2.putText(frame, f'{class_names[i]}: {count}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def process_frame(model, frame, deepsort, frame_count):
    results = model(frame)
    detections = []
    classes = []
    confidences = []  # List to store confidence scores for each detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.numpy()[0]
            w, h = x2 - x1, y2 - y1
            detection = np.array([x1, y1, w, h])
            detections.append(detection)
            classes.append(int(box.cls))  # Assuming box.cls provides the class id
            confidences.append(float(box.conf))  # Assuming box.conf provides the confidence score
    return detections, classes, confidences

def run_deepsort_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    deepsort = DeepSort()
    model = YOLO('30.pt')
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections, classes, confidences = process_frame(model, frame, deepsort, frame_count)
        deepsort.update(detections, frame_count, classes)

        deepsort.draw_paths(frame)
        deepsort.count_vehicles_passing_red_line()
        deepsort.print_vehicle_count(frame)

        for i, tracker in enumerate(deepsort.trackers):
            x, y, w, h = tracker.detection
            class_id = tracker.class_id
            confidence = confidences[i] if i < len(confidences) else 0  # Get confidence level for the class
            class_names = ["Bicycle", "Bus", "Cars", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"]
            label = f'{class_names[class_id]}: {confidence:.2f}'
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'k.mp4'
    run_deepsort_on_video(video_path)
