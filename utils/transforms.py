import numpy as np
from geometry_msgs.msg import Pose


def euler_to_quaternion_angles(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def get_pose(position, orientation):
    grasp_pose = Pose()
    grasp_pose.position.x = position[0]
    grasp_pose.position.y = position[1]
    grasp_pose.position.z = position[2]
    grasp_pose.orientation.x = orientation[0]
    grasp_pose.orientation.y = orientation[1]
    grasp_pose.orientation.z = orientation[2]
    grasp_pose.orientation.w = orientation[3]

    return grasp_pose
