#!/usr/bin/env python3

import rosbag
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import cv2
import yaml
from my_package.srv import KalmanFilter, KalmanFilterRequest
from detection_yolo import detect_objects_ret_cords
from sensor_msgs import point_cloud2
from transformations import transform, load_camera_calibration

camera_dict = load_camera_calibration()

# Define camera to LiDAR topic mappings based on the provided associations
CAMERA_TO_LIDAR_MAP = {
    '/NLBU/image_raw': '/ouster_ll/points',
    '/NLFU/image_raw': '/ouster/points',
    '/NMFU/image_raw': '/ouster/points',
    '/NRBU/image_raw': '/ouster_rl/points',
    '/NRFU/image_raw': '/ouster/points',
    '/SMBU/image_raw': ['/ouster_ll/points', '/ouster_rl/points']
}

TOPIC_TO_CALIB_MAP = {
    '/NLBU/image_raw': "nlbu_camera",
    '/NLFU/image_raw': 'nlfu_camera',
    '/NMFU/image_raw': 'nmfu_camera',
    '/NRBU/image_raw': 'nrbu_camera',
    '/NRFU/image_raw': 'nrfu_camera',
    '/SMBU/image_raw': 'smbu_camera',
    '/ouster_ll/points': 'bl_lidar',
    '/ouster_rl/points': 'br_lidar',
    '/ouster/points': 'f_lidar'
}

# Load calibration data
def load_calibration_file(calib_file):
    with open(calib_file, 'r') as file:
        calib_data = yaml.safe_load(file)
    return calib_data

calib_data = load_calibration_file('calib.yaml')

def process_bag_data(bag_file):
    bag = rosbag.Bag(bag_file)
    bridge = CvBridge()
    
    camera_messages = {topic: [] for topic in CAMERA_TO_LIDAR_MAP.keys()}
    lidar_messages = {topic: [] for topic in set(topic for topics in CAMERA_TO_LIDAR_MAP.values() for topic in (topics if isinstance(topics, list) else [topics]))}
    
    for topic, msg, t in bag.read_messages(topics=list(camera_messages.keys()) + list(lidar_messages.keys())):
        if topic in camera_messages:
            camera_messages[topic].append((msg, t))
        if topic in lidar_messages:
            lidar_messages[topic].append((msg, t))

    print("Finished reading messages")
    
    bag.close()
    return camera_messages, lidar_messages

def call_kalman_service(x, y, z):
    # Initialize ROS node
    rospy.init_node('kalman_client', anonymous=True)
    
    # Wait for the service to become available
    rospy.wait_for_service('kalman_filter')

    try:
        # Create a service proxy
        kalman_service = rospy.ServiceProxy('kalman_filter', KalmanFilter)
        
        # Call the service
        response = kalman_service(x, y, z)
        
        # Return the filtered position
        return response.x, response.y, response.z
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return x, y, z  # Return original values in case of failure

def synchronize_data(camera_messages, lidar_messages):
    synchronized_data = []
    for cam_topic, cam_msgs in camera_messages.items():
        lidar_topics = CAMERA_TO_LIDAR_MAP[cam_topic]
        if isinstance(lidar_topics, str):
            lidar_topics = [lidar_topics]
        
        lidar_msgs_list = {topic: lidar_messages[topic] for topic in lidar_topics if topic in lidar_messages}
        
        for cam_msg, cam_time in cam_msgs:
            closest_lidar_msgs = []
            for lidar_topic, lidar_msgs in lidar_msgs_list.items():
                if lidar_msgs:
                    closest_msg = min(lidar_msgs, key=lambda x: abs((x[1] - cam_time).to_sec()))
                    closest_lidar_msgs.append((closest_msg[0], lidar_topic))
            
            synchronized_data.append((cam_msg, cam_topic, closest_lidar_msgs))

    print("Synchronization complete")
    return synchronized_data

def point_cloud_to_numpy(point_cloud_msg):
    pc_data = point_cloud2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_array = np.array(list(pc_data))
    return pc_array

def display_frames_continuously(synchronized_data):
    bridge = CvBridge()
    
    for i, (cam_msg, cam_topic, lidar_msgs) in enumerate(synchronized_data):
        
        camera_id = TOPIC_TO_CALIB_MAP[cam_topic]
        image = bridge.imgmsg_to_cv2(cam_msg, "bgr8")
        
        intrinsic = camera_dict[camera_id]["intrinsic"]
        distortion = camera_dict[camera_id]["distortion"]
        image = cv2.undistort(image, intrinsic, distortion)
        
        detections = detect_objects_ret_cords(image)
        
        projected_points_dict = {label: [] for label in detections.keys()}
        
        for lidar_msg, lidar_topic in lidar_msgs:
            lidar_id = TOPIC_TO_CALIB_MAP[lidar_topic]
            points = point_cloud_to_numpy(lidar_msg)
            
            for point in points:
                u, v, z = transform(point, lidar_id, camera_id)
                if u is not None and v is not None:
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        for label, bboxes in detections.items():
                            for bbox in bboxes:
                                xmin, ymin, xmax, ymax = bbox
                                if xmin <= u <= xmax and ymin <= v <= ymax:
                                    # Call Kalman service for 3D coordinates
                                    filtered_x, filtered_y, filtered_z = call_kalman_service(u, v, z)
                                    projected_points_dict[label].append((filtered_x, filtered_y))
                                    break
        
        for label, bboxes in detections.items():
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                for (u, v) in projected_points_dict[label]:
                    cv2.circle(image, (int(u), int(v)), 2, (255, 0, 0), -1)
        
        cv2.imshow('Camera Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Process the bag file
bag_file = 'filtered.bag'
print(f"Starting data processing for {bag_file}")

camera_messages, lidar_messages = process_bag_data(bag_file)
synchronized_data = synchronize_data(camera_messages, lidar_messages)
display_frames_continuously(synchronized_data)

print("Data processing and display completed.")
