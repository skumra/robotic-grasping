import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from hardware.camera import RealSenseCamera


class Calibration:
    def __init__(self,
                 cam_id,
                 calib_grid_step,
                 checkerboard_offset_from_tool,
                 workspace_limits
                 ):
        self.calib_grid_step = calib_grid_step
        self.checkerboard_offset_from_tool = checkerboard_offset_from_tool

        # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.workspace_limits = workspace_limits

        self.camera = RealSenseCamera(device_id=cam_id)
        
        self.measured_pts = []
        self.observed_pts = []
        self.observed_pix = []
        self.world2camera = np.eye(4)

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        self.move_completed = os.path.join(homedir, "move_completed.npy")
        self.tool_position = os.path.join(homedir, "tool_position.npy")

    @staticmethod
    def _get_rigid_transform(A, B):
        """
        Estimate rigid transform with SVD (from Nghia Ho)
        """
        assert len(A) == len(B)

        N = A.shape[0]  # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def _get_rigid_transform_error(self, z_scale):
        """
        Calculate the rigid transform RMS error

        :return RMS error
        """
        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = np.squeeze(self.observed_pts[:, 2:] * z_scale)
        observed_x = np.multiply(np.squeeze(self.observed_pix[:, [0]]) - self.camera.intrinsics.ppx,
                                 observed_z / self.camera.intrinsics.fx)
        observed_y = np.multiply(np.squeeze(self.observed_pix[:, [1]]) - self.camera.intrinsics.ppy,
                                 observed_z / self.camera.intrinsics.fy)

        new_observed_pts = np.asarray([observed_x, observed_y, observed_z]).T

        # Estimate rigid transform between measured points and new observed points
        R, t = self._get_rigid_transform(np.asarray(self.measured_pts), np.asarray(new_observed_pts))
        t.shape = (3, 1)
        self.world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(self.measured_pts)) + np.tile(t, (1, self.measured_pts.shape[0]))
        error = np.transpose(registered_pts) - new_observed_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / self.measured_pts.shape[0])
        return rmse

    def _generate_grid(self):
        """
        Construct 3D calibration grid across workspace

        :return calibration grid points
        """
        gridspace_x = np.linspace(self.workspace_limits[0][0], self.workspace_limits[0][1],
                                  1 + (self.workspace_limits[0][1] - self.workspace_limits[0][
                                      0]) / self.calib_grid_step)
        gridspace_y = np.linspace(self.workspace_limits[1][0], self.workspace_limits[1][1],
                                  1 + (self.workspace_limits[1][1] - self.workspace_limits[1][
                                      0]) / self.calib_grid_step)
        gridspace_z = np.linspace(self.workspace_limits[2][0], self.workspace_limits[2][1],
                                  1 + (self.workspace_limits[2][1] - self.workspace_limits[2][
                                      0]) / self.calib_grid_step)
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts
        
    def run(self):
        # Connect to camera
        self.camera.connect()
        logging.debug(self.camera.intrinsics)

        logging.info('Collecting data...')

        calib_grid_pts = self._generate_grid()

        logging.info('Total grid points: ', calib_grid_pts.shape[0])

        for tool_position in calib_grid_pts:
            logging.info('Requesting move to tool position: ', tool_position)
            np.save(self.tool_position, tool_position)
            np.save(self.move_completed, 0)
            while not np.load(self.move_completed):
                time.sleep(0.1)
            # Wait for robot to be stable
            time.sleep(2)

            # Find checkerboard center
            checkerboard_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_bundle = self.camera.get_image_bundle()
            camera_color_img = image_bundle['rgb']
            camera_depth_img = image_bundle['aligned_depth']
            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)

                # Get observed checkerboard center 3D point in camera space
                checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
                checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                checkerboard_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                             checkerboard_z / self.camera.intrinsics.fx)
                checkerboard_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                             checkerboard_z / self.camera.intrinsics.fy)
                if checkerboard_z == 0:
                    continue

                # Save calibration point and observed checkerboard center
                self.observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
                # tool_position[2] += self.checkerboard_offset_from_tool
                tool_position = tool_position + self.checkerboard_offset_from_tool

                self.measured_pts.append(tool_position)
                self.observed_pix.append(checkerboard_pix)

                # Draw and display the corners
                vis = cv2.drawChessboardCorners(bgr_color_data, (1, 1), corners_refined[4, :, :], checkerboard_found)
                # cv2.imwrite('%06d.png' % len(self.measured_pts), vis)
                cv2.imshow('Calibration', vis)
                cv2.waitKey(10)
            else:
                logging.info('Checker board not found')

        self.measured_pts = np.asarray(self.measured_pts)
        self.observed_pts = np.asarray(self.observed_pts)
        self.observed_pix = np.asarray(self.observed_pix)

        # Optimize z scale w.r.t. rigid transform error
        logging.info('Calibrating...')
        z_scale_init = 1
        optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
        camera_depth_offset = optim_result.x

        # Save camera optimized offset and camera pose
        logging.info('Saving...')
        np.savetxt('saved_data/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
        rmse = self._get_rigid_transform_error(camera_depth_offset)
        logging.info('RMSE: ', rmse)
        camera_pose = np.linalg.inv(self.world2camera)
        np.savetxt('saved_data/camera_pose.txt', camera_pose, delimiter=' ')
        logging.info('Done.')
