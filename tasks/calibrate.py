import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from interfaces.camera2 import Camera
from interfaces.robot import Robot


class Calibration:
    def __init__(self,
                 robot_ip='127.0.0.1',
                 robot_port=1000,
                 calib_grid_step=0.05,
                 checkerboard_offset_from_tool=[0, -0.13, 0.02],
                 tool_orientation=[-np.pi / 2, 0, 0]
                 ):
        self.calib_grid_step = calib_grid_step
        self.checkerboard_offset_from_tool = checkerboard_offset_from_tool
        self.tool_orientation = tool_orientation

        self.workspace_limits = np.asarray([[0.3, 0.748], [0.05, 0.4], [-0.2, -0.1]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

        self.camera = Camera()
        self.robot = Robot(robot_ip, robot_port, self.workspace_limits)

        self.measured_pts = []
        self.observed_pts = []
        self.observed_pix = []

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
        observed_z = self.observed_pts[:, 2:] * z_scale
        observed_x = np.multiply(self.observed_pix[:, [0]] - self.camera.intrinsics[0][2],
                                 observed_z / self.camera.intrinsics[0][0])
        observed_y = np.multiply(self.observed_pix[:, [1]] - self.camera.intrinsics[1][2],
                                 observed_z / self.camera.intrinsics[1][1])
        new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

        # Estimate rigid transform between measured points and new observed points
        R, t = self._get_rigid_transform(np.asarray(self.measured_pts), np.asarray(new_observed_pts))
        t.shape = (3, 1)
        world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

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
        # Move robot to home pose
        print('Moving to start position...')
        self.robot.go_home()
        self.robot.open_gripper()

        # Make robot gripper point upwards
        self.robot.move_joints([-np.pi, -np.pi / 2, np.pi / 2, 0, np.pi / 2, np.pi])

        # Move robot to each calibration point in workspace
        print('Collecting data...')

        calib_grid_pts = self._generate_grid()

        for tool_position in calib_grid_pts:
            self.robot.move_to(tool_position, self.tool_orientation)
            time.sleep(1)

            # Find checkerboard center
            checkerboard_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            camera_color_img, camera_depth_img = self.camera.get_data()
            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)

                # Get observed checkerboard center 3D point in camera space
                checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
                checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                checkerboard_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics[0][2],
                                             checkerboard_z / self.camera.intrinsics[0][0])
                checkerboard_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics[1][2],
                                             checkerboard_z / self.camera.intrinsics[1][1])
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
                cv2.imwrite('%06d.png' % len(self.measured_pts), vis)
                cv2.imshow('Calibration', vis)
                cv2.waitKey(10)

        # Move robot back to home pose
        self.robot.go_home()

        measured_pts = np.asarray(self.measured_pts)
        observed_pts = np.asarray(self.observed_pts)
        observed_pix = np.asarray(self.observed_pix)
        world2camera = np.eye(4)

        # Optimize z scale w.r.t. rigid transform error
        print('Calibrating...')
        z_scale_init = 1
        optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
        camera_depth_offset = optim_result.x

        # Save camera optimized offset and camera pose
        print('Saving...')
        np.savetxt('camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
        self._get_rigid_transform_error(camera_depth_offset)
        camera_pose = np.linalg.inv(world2camera)
        np.savetxt('camera_pose.txt', camera_pose, delimiter=' ')
        print('Done.')

        # ---------------------------------------------

        np.savetxt('measured_pts.txt', np.asarray(measured_pts), delimiter=' ')
        np.savetxt('observed_pts.txt', np.asarray(observed_pts), delimiter=' ')
        np.savetxt('observed_pix.txt', np.asarray(observed_pix), delimiter=' ')
        measured_pts = np.loadtxt('measured_pts.txt', delimiter=' ')
        observed_pts = np.loadtxt('observed_pts.txt', delimiter=' ')
        observed_pix = np.loadtxt('observed_pix.txt', delimiter=' ')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(measured_pts[:,0],measured_pts[:,1],measured_pts[:,2], c='blue')

        print(camera_depth_offset)
        R, t = self._get_rigid_transform(np.asarray(measured_pts), np.asarray(observed_pts))
        t.shape = (3,1)
        camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
        camera2robot = np.linalg.inv(camera_pose)
        t_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(observed_pts)) + np.tile(camera2robot[0:3,3:],(1,observed_pts.shape[0])))

        ax.scatter(t_observed_pts[:,0],t_observed_pts[:,1],t_observed_pts[:,2], c='red')

        new_observed_pts = observed_pts.copy()
        new_observed_pts[:,2] = new_observed_pts[:,2] * camera_depth_offset[0]
        R, t = self._get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
        t.shape = (3,1)
        camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
        camera2robot = np.linalg.inv(camera_pose)
        t_new_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(new_observed_pts)) + np.tile(camera2robot[0:3,3:],(1,new_observed_pts.shape[0])))

        ax.scatter(t_new_observed_pts[:,0],t_new_observed_pts[:,1],t_new_observed_pts[:,2], c='green')

        plt.show()


if __name__ == '__main__':
    calibration = Calibration()
    calibration.run()
