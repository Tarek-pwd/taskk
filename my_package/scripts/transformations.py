#!/usr/bin/env python3
import yaml
import numpy as np
from collections import defaultdict

def load_camera_calibration():
    cameras = defaultdict(dict)
    camera_names = ['nlfu_camera', 'nmfu_camera', 'nmfd_camera', 'nrfu_camera', 'slfu_camera', 'srfu_camera', 'nlbu_camera', 'nrbu_camera', 'smbu_camera']
    params = ['intrinsic', 'from_ego', 'distortion']

    for camera in camera_names:
        per_cam_dict = {}
        for param in params:
            per_cam_dict[param] = np.array(data[camera][param])
        cameras[camera] = per_cam_dict
    return cameras

def load_Lidar_calibration():
    Lidar_names = ["br_lidar", "bl_lidar", "f_lidar"]
    Lidars = {}
    for lidar in Lidar_names:
        Lidars[lidar] = np.array(data[lidar]["from_ego"])
    return Lidars

def transform(point, lidar_id, cam_id):
    out = transform_lidar_to_ego(point, lidar_id)
    if out is None:
        return None
    out = transform_ego_to_camera(out, cam_id)
    u, v, z= project_to_image(out, cam_id)
    # print("TRANSFORMED TO", u, v)
    return u, v, z 

def transform_lidar_to_ego(point, lidar_id):
    try:
        lidar_to_ego = np.linalg.inv(lidars[lidar_id])
    except np.linalg.LinAlgError as e:
        print(f"Matrix inversion error for {lidar_id}: {e}")
        return None

    point_homogeneous = np.append(point, 1)  # Add homogeneous coordinate
    ego_point_homogeneous = np.dot(lidar_to_ego, point_homogeneous)
    return ego_point_homogeneous[:3]  # Return 3D point in ego coordinates

def transform_ego_to_camera(point_hom, camera_id):
    camera_to_ego = cameras[camera_id]['from_ego']
    camera_point_homogeneous = np.dot(camera_to_ego, np.append(point_hom, 1))
    return camera_point_homogeneous[:3]  # Return 3D point in camera coordinates

def project_to_image(camera_point, camera_id):
    intrinsic = cameras[camera_id]['intrinsic']
    x, y, z = camera_point
    
    if z == 0:
        print("DIVISION BY ZERO -- PROBABLY LIDAR F")
        return None
    
    # Convert to 2D image plane coordinates
    u = (intrinsic[0, 0] * x + intrinsic[0, 1] * y + intrinsic[0, 2] * z) / z
    v = (intrinsic[1, 0] * x + intrinsic[1, 1] * y + intrinsic[1, 2] * z) / z
    
    return int(u), int(v) ,int(z)

file_path = "calib.yaml"
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

cameras = load_camera_calibration()
lidars = load_Lidar_calibration()

print(load_camera_calibration()["nlbu_camera"])
# Example usage:
# transform([1, 2, 3], "bl_lidar", "nlbu_camera")
