import copy

from geometry_msgs.msg import Pose

from interfaces.camera import RealSenseCamera
from interfaces.robot import Robot
from inference.grasp_generator import GraspGenerator


class PickAndPlace:
    def __init__(
            self,
            saved_model,
            robot_ip='127.0.0.1',
            robot_port=1000,
            cam_id=830112070066,
            hover_distance=0.15
    ):
        self._hover_distance = hover_distance  # in meters
        self.saved_model = saved_model

        self.camera = RealSenseCamera(device_id=cam_id)
        self.robot = Robot(robot_ip, robot_port)
        self.grasp_generator = GraspGenerator(saved_model=saved_model, camera=self.camera)

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

    def run(self):
        # Connect to camera
        self.camera.connect()

        # Connect to robot
        self.robot.connect()

        # Load model
        print('Loading model... ')
        self.grasp_generator.load_model()

        # Move robot to home pose
        print('Moving to start position...')
        self.robot.go_home()
        self.robot.open_gripper()
