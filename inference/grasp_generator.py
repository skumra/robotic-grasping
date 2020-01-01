import numpy as np
import torch

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.transforms import euler_to_quaternion


class GraspGenerator:
    def __init__(self, saved_model, camera):
        self.saved_model = saved_model
        self.camera = camera
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def load_model(self):
        self.model = torch.load(self.saved_model)
        self.device = torch.device("cuda:0")

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasp = detect_grasps(q_img, ang_img, width_img)[0]

        # Get grasp position from model output
        pos_z = depth_img[grasp.center[0]][grasp.center[1]] * self.cam_depth_scale
        pos_x = np.multiply(grasp.center[1] - self.camera.cam_intrinsics[0][2], pos_z / self.camera.cam_intrinsics[0][0])
        pos_y = np.multiply(grasp.center[0] - self.camera.cam_intrinsics[1][2], pos_z / self.camera.cam_intrinsics[1][1])
        if pos_z == 0:
            return
        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)

        # Convert camera to robot coordinates
        camera2robot = self.cam_pose
        target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]

        target_position = target_position[0:3, 0]

        # Get grasp angle from model output
        target_orientation = euler_to_quaternion(0, 0, grasp.angle)

        # Calculate grasp pose
        grasp_pose = np.concatenate((target_position, target_orientation))

        return grasp_pose
