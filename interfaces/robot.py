import numpy as np


class Robot:
    def __init__(self, robot_ip, robot_port, workspace_limits):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.workspace_limits = workspace_limits

        self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

    def connect(self):
        """ Establish connection with the robot """
        pass

    def close_gripper(self):
        """ Close robot gripper """
        pass

    def open_gripper(self):
        """ Open robot gripper """
        pass

    def move_to(self, tool_position, tool_orientation):
        """ Move robot in cartesian space"""
        pass

    def move_joints(self, joint_configuration):
        """ Move robot in joint space """
        pass

    def go_home(self):
        """ Move robot to home position """
        self.move_joints(self.home_joint_config)
