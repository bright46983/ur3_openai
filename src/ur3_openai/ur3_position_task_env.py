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
        
        # limit
        self.num_joints = 6
        self.joint_lower_limit = [-np.pi,-np.pi/2,-np.pi/2,-np.pi,-np.pi,-np.pi]
        self.joint_upper_limit = [np.pi,np.pi/2,np.pi/2,np.pi,np.pi,np.pi]

        self.tran_upper_limit = [0.05,0.05,0.05,0,0,0] #  [x,y,z,roll,pitch,yaw]
        self.tran_lower_limit = [-0.05,-0.05,-0.05,-0,-0,-0] #  [x,y,z,roll,pitch,yaw]

        self.workspace_upper_limit = [0.5,0.7,0.6]#[x,y,z] ref on 'base_link'
        self.workspace_lower_limit = [-0.5,0.1,0.05]#y pointing to table, z pointing up

        # position goal
        self.goal = self.get_random_workspace_ee_position()
        self.goal_tolerance = 0.1
        self.cumulative_reward = 0

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
        self.is_ur3_collided = False
        self.goal = self.get_random_workspace_ee_position()
        self.cumulative_reward = 0


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
        # get ee pose 
        ee_pose = self.arm.end_effector(rot_type='euler',tip_link='gripper_tip_link') #[x, y, z, roll, pitch, yaw] np.array
        is_collided = float(self.is_ur3_collided) # 1.0 or 0.0

        observations = np.block([ee_pose[:3], is_collided])  # [ee_x,ee_y,ee_z,is_colided]

        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        # TODO
        is_near_goal = self.check_near_goal( observations[:3])
        if is_near_goal:
            rospy.loginfo("Reach Current Goal")

        is_colided = bool(observations[-1])
        if is_colided:
            rospy.logwarn("UR3 is collided")

        done = is_near_goal or is_colided
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        # TODO
        reward = -1
        is_near_goal = self.check_near_goal( observations[:3])
        is_colided = bool(observations[-1])

        if is_near_goal:
            reward += 100
        if is_colided:
            reward -= 20
        
        self.cumulative_reward += reward
        rospy.loginfo("Reward: {}".format(reward))
        rospy.loginfo("Cumulative reward: {}".format(self.cumulative_reward))
        return reward
        
    # Internal TaskEnv Methods
    def get_random_workspace_ee_position(self):
        """
        Get a random position in the workspace.
        """
        # TODO
        return np.random.uniform(low=np.array(self.workspace_lower_limit), high=np.array(self.workspace_upper_limit), size=(1,3))

    def check_near_goal(self, ee_pose):
        return np.linalg.norm(self.goal.reshape(1,3) - ee_pose.reshape(1,3))  < self.goal_tolerance
