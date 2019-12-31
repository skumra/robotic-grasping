import copy

from geometry_msgs.msg import Pose

import interfaces.robot


class PickAndPlace:
    def __init__(
            self,
            robot: interfaces.robot.Robot,
            hover_distance=0.15
    ):
        self.robot = robot
        self._hover_distance = hover_distance  # in meters

    def _approach(self, pose):
        """
        Approach with a pose the hover-distance above the requested pose
        """
        approach = copy.deepcopy(pose)
        approach.position.z = approach.position.z + self._hover_distance
        self.robot.move_to(approach)

    def _servo_to_pose(self, pose):
        """
        Servo down to pose
        """
        self.robot.move_to(pose)

    def _retract(self):
        """
        Retract up from current pose
        """
        # retrieve current pose from endpoint
        current_pose = self.robot.current_pose()
        pose = Pose()
        pose.position.x = current_pose['position'].x
        pose.position.y = current_pose['position'].y
        current_pose.position.z = current_pose['position'].z + self._hover_distance
        pose.orientation.x = current_pose['orientation'].x
        pose.orientation.y = current_pose['orientation'].y
        pose.orientation.z = current_pose['orientation'].z
        pose.orientation.w = current_pose['orientation'].w

        # servo up from current pose
        self.robot.move_to(pose)

    def pick(self, pose):
        """
        Pick from given pose
        """
        # open the gripper
        self.robot.open_gripper()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.robot.close_gripper()
        # retract to clear object
        self._retract()

    def place(self, pose):
        """
        Place to given pose
        """
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.robot.open_gripper()
        # retract to clear object
        self._retract()
