import numpy as np
import struct
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface


class Robot:
    def __init__(self, robot_ip, robot_port, workspace_limits,limb,verbose=True):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.workspace_limits = workspace_limits

        self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self._limb_name = limb # string
        self._verbose = verbose
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)

    def _ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints    

    def connect(self):
        """ Establish connection with the robot """
        # verify robot is enabled
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        init_state = rs.state().enabled
        print("Enabling robot... ")
        rs.enable()

    def close_gripper(self):
        """ Close robot gripper """
        self._gripper.close()
        rospy.sleep(1.0)

    def open_gripper(self):
        """ Open robot gripper """
        self._gripper.open()
        rospy.sleep(1.0)

    def move_to(self, pose):
        """ Move robot in cartesian space """
        joint_angles = self._ik_request(pose)
        self.move_joints(joint_angles)

    def move_joints(self, joint_angles):
        """ Move robot in joint space """
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def current_pose(self):
        """ Return current pose from endpoint """
        current_pose = self._limb.endpoint_pose()
        return current_pose

    def go_home(self):
        """ Move robot to home position """
        self.move_joints(self.home_joint_config)
