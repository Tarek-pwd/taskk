#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from collections import defaultdict

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# # Initialize Kalman Filter for each object with position and velocity tracking
# class KalmanTracker:
#     def __init__(self):
#         # Define the Kalman Filter
#         # State vector [x, y, vx, vy] where (x, y) is the position and (vx, vy) is the velocity
#         self.kf = cv2.KalmanFilter(4, 2)
#         self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
#                                               [0, 1, 0, 0]], np.float32)
#         self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
#                                              [0, 1, 0, 1],
#                                              [0, 0, 1, 0],
#                                              [0, 0, 0, 1]], np.float32)
#         self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

#         # Initialize the state vector
#         self.last_detection = None

#     def predict(self):
#         # Predict the next position and velocity
#         predicted = self.kf.predict()
#         return predicted[0], predicted[1], predicted[2], predicted[3]

#     def update(self, detection):
#         if self.last_detection is None:
#             # Initialize the filter with the first detection, assuming initial velocity is 0
#             self.kf.statePre = np.array([[detection[0]], [detection[1]], [0], [0]], np.float32)
#             self.kf.statePost = self.kf.statePre
#             self.last_detection = detection
#         else:
#             # Update the filter with the new detection (position)
#             self.kf.correct(np.array([[detection[0]], [detection[1]]], np.float32))
#             self.last_detection = detection

# # Initialize a tracker dictionary to keep track of objects
# trackers = {}


def detect_objects_ret_cords(img):
    results = model(img)
    detections = results.pandas().xyxy[0]
    detection_dict = defaultdict(list)  # Use defaultdict to initialize lists
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        detection_dict[label].append((xmin, ymin, xmax, ymax))
    
    return detection_dict




# def detect_objects(img):
#     global trackers
    
#     results = model(img)
#     # Get detection results as DataFrame
#     detections = results.pandas().xyxy[0]
    
#     for _, row in detections.iterrows():
#         xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#         label = row['name']
#         confidence = row['confidence']
        
#         # Calculate the center point of the detection
#         center_x = (xmin + xmax) // 2
#         center_y = (ymin + ymax) // 2
        
#         # Initialize a tracker if it doesn't exist
#         if label not in trackers:
#             trackers[label] = KalmanTracker()
        
#         # Update the tracker with the new detection
#         trackers[label].update((center_x, center_y))
        
#         # Predict the new position and velocity
#         predicted_x, predicted_y, predicted_vx, predicted_vy = trackers[label].predict()
        
#         # Convert predicted center back to bounding box coordinates
#         box_width = xmax - xmin
#         box_height = ymax - ymin
#         predicted_xmin = int(predicted_x - box_width // 2)
#         predicted_ymin = int(predicted_y - box_height // 2)
#         predicted_xmax = int(predicted_x + box_width // 2)
#         predicted_ymax = int(predicted_y + box_height // 2)
        
#         # Draw a rectangle around the predicted bounding box
#         cv2.rectangle(img, (predicted_xmin, predicted_ymin), (predicted_xmax, predicted_ymax), (0, 255, 0), 2)
#         cv2.putText(img, f'{label} {confidence:.2f}', (predicted_xmin, predicted_ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Draw the predicted center
#         cv2.circle(img, (int(predicted_x), int(predicted_y)), 3, (0, 0, 255), -1)
        
#     return img
