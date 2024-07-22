"""Template file used that can be used to create a new Robot environment. It contains a
Python class that specifies the robot to use on the task. It provides the complete
integration between the Gazebo simulation of the robot and the gymnasium library, so
obtaining **SENSOR** information from the robot or sending **ACTIONS** to it are
transparent to the gymnasium library and you, the developer. For more information, see
the `ros_gazebo_gym <https://rickstaa.dev/ros-gazebo-gym>`_ documentation.

Source code
-----------

.. literalinclude:: ../../../../../templates/template_my_robot_env.py
   :language: python
   :linenos:
   :lines: 16-
"""
import rospy
import numpy as np
from ros_gazebo_gym.robot_gazebo_env import RobotGazeboEnv
from ur_control.arm import Arm
from ur_control.constants import GripperType, JOINT_ORDER
from geometry_msgs.msg import WrenchStamped



class UR3Env(RobotGazeboEnv):
    """Superclass for all Robot environments."""

    def __init__(self):
        """Initializes a new Robot environment."""
        # Setup internal robot environment variables (controllers, namespace etc.).
        # NOTE: If the controllers_list is not set all the currently running controllers
        # will be reset by the ControllersList class.
        # TODO: Change the controller list and robot namespace.
        self.controllers_list = [
           'joint_state_controller',
           'scaled_pos_joint_traj_controller',
           'gripper_controller']
        
        self.robot_name_space = ""
        reset_controls_bool = True

        # UR3 Specific
        self.model_name = "robot"
        self.arm = Arm(gripper_type=GripperType.GENERIC)
        self.joint_initial_positions = self.arm.joint_angles()
        self.joint_states = None
        
        self.wrench = None
        self.collision_threshold = 200
        self.is_ur3_collided = False

        rospy.Subscriber('/wrench',WrenchStamped, self.wrench_cb)
        rospy.Subscriber('/joint_states',WrenchStamped, self.js_cb)

        

        while not rospy.is_shutdown() and (self.joint_states == None or self.wrench is None):
            rospy.logwarn("Waiting for joint state to arrive")
        self.joint_states_inititial = self.joint_states
        # Initialize the gazebo environment.
        super(UR3Env, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=reset_controls_bool,
            reset_robot_pose=True,
            reset_world_or_sim="WORLD"
        )

        

    ################################################
    # Overload Gazebo env virtual methods ##########
    ################################################
    # NOTE: Methods that need to be implemented as they are called by the robot and
    # Gazebo environments.
    def _check_all_systems_ready(self):
        """Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO: Implement the logic that checks if all sensors and actuators are ready.
        
        while not rospy.is_shutdown():
            rospy.logwarn("Waiting for joint state to arrive")
            print((rospy.Time.now() - self.js_timestamp).to_sec())
            if  abs(rospy.Time.now() - self.js_timestamp).to_sec() < 0.2 and  abs(rospy.Time.now() - self.wrench_timestamp).to_sec() < 0.2:
                break
            rospy.sleep(0.1)

        self.arm = Arm(gripper_type=GripperType.GENERIC)

        return True

    ################################################
    # Robot environment internal methods ###########
    ################################################
    # NOTE: Here you can add additional helper methods that are used in the robot env.
    def wrench_cb(self,msg):
        """
        Callback for the wrench topic.
        """
        self.wrench = msg
        self.wrench_timestamp = rospy.Time.now()
        if not self.is_ur3_collided:
            self.is_ur3_collided  = self._is_collided()

    def js_cb(self, msg):
        self.joint_states = msg 
        self.js_timestamp = rospy.Time.now()

    def _is_collided(self):
        """
        Checks if there are any collisions in the environment.
        """
        if self.wrench:
            fx = abs(self.wrench.wrench.force.x)
            fy = abs(self.wrench.wrench.force.y)
            fz = abs(self.wrench.wrench.force.z)

            if fx > self.collision_threshold or fy > self.collision_threshold or fz > self.collision_threshold:
                rospy.logerr("Collision detected!!!!")
                return True
            else:
                return False

        else:
            rospy.logerr("Wrench msg not arrived")
            return False
        
    
        

    ################################################
    # Robot env main methods #######################
    ################################################
    # NOTE: Contains methods that the task environment will need.
    def move_joints(self, ang_list, wait, target_time=10.0):
        """
        Moves the robot to a new joint configuration.
        :param ang_list(list): list of new joint configurations [q1,q2,q3]
        :param wait(bool): Whether to wait for arm to arrive at new joint configuration if not, 
        controller's result will always be DONE
        :param target_time(float): target time to reach the new joint configuration in seconds (default: 10.0) 
        if not, controller will return FAIL
        """
        q = ang_list
        res = self.arm.set_joint_positions(positions=q, wait=wait, target_time=target_time)

    def move_ee(self, pose, wait= True, target_time=5.0):
        """
        Moves the end-effector according to displacement
        :param pose: pose value  [x,y,z,aw, ax, ay, az]
        :param wait(bool): Whether to wait for arm to arrive at new joint configuration if not, 
        controller's result will always be DONE
        :param target_time(float): target time to reach the new joint configuration in seconds (default: 10.0) 
        if not, controller will return FAIL
        """
        print(pose)
        pose = np.array(pose)
        res = self.arm.set_target_pose(pose=pose, wait=wait, target_time=target_time)
        rospy.logwarn("move_ee...")

    def move_ee_relative(self, trans, wait= True, target_time=5.0,relative_to_tcp = True):
        """
        Moves the end-effector according to displacement
        :param trans: transformation value  [x,y,z,roll, pitch, yaw]
        :param relative_to_tcp(bool): relative to lastest ee frame (True) or to base frame(False)
        :param wait(bool): Whether to wait for arm to arrive at new joint configuration if not, 
        controller's result will always be DONE
        :param target_time(float): target time to reach the new joint configuration in seconds (default: 10.0) 
        if not, controller will return FAIL
        """
        
        res = self.arm.move_relative(transformation=trans, wait=wait, target_time=target_time, relative_to_tcp = relative_to_tcp)
        rospy.logwarn("move_ee_relative...")