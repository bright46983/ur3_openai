import rospy
import numpy as np
from ur3_openai import ur3_gazebo_env
from openai_ros import robot_gazebo_env
import tf
from ur_control import transformations, traj_utils, conversions
from ur_control.arm import Arm
from ur_control.constants import GripperType, JOINT_ORDER
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped



class UR3Env(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['joint_state_controller','scaled_pos_joint_traj_controller','gripper_controller'] # for ros_control?
        # ,'shoulder_pan_joint'
        #                          ,'shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'
        self.robot_name_space = ""

        reset_controls_bool = True # not sure

        self.arm = Arm(gripper_type=GripperType.GENERIC)
        self.joint_initial_positions = self.arm.joint_angles()

        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(UR3Env, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool,
                                                reset_world_or_sim="WORLD") # only rest object position
        
        self.gazebo.unpauseSim()

        
        
        # rospy.sleep(5)
        #self._check_all_sensors_ready()

        # ur3 controller module
        
        # Subscibers for observation 
        self.wrench = None
        self.collision_threshold = 200
        self.is_ur3_collided = False
        rospy.Subscriber('/wrench',WrenchStamped, self.wrench_cb)
        
        rospy.logwarn("Finished UR3Env INIT...")
        

    # Methods needed by the RobotGazeboEnv Ex. sub cb, check method, ...
    # ----------------------------
    def wrench_cb(self,msg):
        """
        Callback for the wrench topic.
        """
        self.wrench = msg
        self.is_ur3_collided  = self._is_collided()

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

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        # check if observation arrived (not implemeted yet)
        return True
        
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
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

