import numpy as np
from gym import spaces
from ur3_openai import ur3_env
from gym.envs.registration import register
from ur_control.constants import GripperType, JOINT_ORDER

import rospy

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
timestep_limit_per_episode = 1000 # Can be any Value

# register(
#         id='UR3Position-v0',
#         entry_point='ur3_openai.ur3_position_task_env:UR3PositionEnv',
#         timestep_limit=timestep_limit_per_episode,
#     )

class UR3PositionEnv(ur3_env.UR3Env):
    def __init__(self):
        
         # others variables
        # self.controllers_type = "joints_position"
        self.controllers_type = "ee_transform"

        self.num_joints = 6
        self.joint_lower_limit = [-np.pi,-np.pi/2,-np.pi/2,-np.pi,-np.pi,-np.pi]
        self.joint_upper_limit = [np.pi,np.pi/2,np.pi/2,np.pi,np.pi,np.pi]

        self.tran_upper_limit = [0.40,0.40,0.40,0,0,0] #  [x,y,z,roll,pitch,yaw]
        self.tran_lower_limit = [-0.40,-0.40,-0.40,-0,-0,-0] #  [x,y,z,roll,pitch,yaw]

        #self.gazebo.unpauseSim()

        # Only variable needed to be set here
        number_actions = rospy.get_param('/my_robot_namespace/n_actions',3)
        if self.controllers_type == 'joints_position':
            self.action_space = self._get_joint_position_action_space()
        elif self.controllers_type == "ee_transform":
            self.action_space = self._get_ee_transform_action_space()
        
        
        # This is the most common case of Box observation type
        # high = numpy.array([
        #     obs1_max_value,
        #     obs12_max_value,
        #     ...
        #     obsN_max_value
        #     ])

        high = np.array([10,10,10]) # assume as ee position    
        self.observation_space = spaces.Box(-high, high)
        
        # Variables that we retrieve through the param server, loded when launch training launch.
        


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(UR3PositionEnv, self).__init__()


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        Got called before reset sim (our case only world)
        """
        # TODO
        rospy.logwarn("Moving joint to initial position ...")
        q = self.joint_initial_positions
        self.move_joints(q, wait=True, target_time = 3.0)


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # TODO
        ...


    def _get_joint_position_action_space(self)-> spaces.Box:
        """
        Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.array(self.joint_lower_limit), high=np.array(self.joint_upper_limit), dtype=np.float32)
    
    def _get_ee_transform_action_space(self)-> spaces.Box:
        """
        Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.array(self.tran_lower_limit), high=np.array(self.tran_upper_limit), dtype=np.float32)

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        # TODO: Move robot
        if self.controllers_type == 'joints_position':
            q = action
            self.move_joints(q, wait=True, target_time = 3.0)
        elif self.controllers_type == "ee_transform":
            tran = action
            self.move_ee_relative(tran,wait=True,target_time= 2.0)
        

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        # TODO
        observations = np.array([1.0, 2.0, 3.0])
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        # TODO
        done = self.is_ur3_collided
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        # TODO
        reward = -1
        return reward
        
    # Internal TaskEnv Methods

