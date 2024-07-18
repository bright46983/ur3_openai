"""Template file used for creating a new task environment. It contains a Python class
that allows specifying the **TASK** that the robot has to learn; for more information,
see the `ros_gazebo_gym <https://rickstaa.dev/ros-gazebo-gym>`_ documentation.

Source code
-----------

.. literalinclude:: ../../../../../templates/template_my_task_env.py
   :language: python
   :linenos:
   :lines: 13-
"""

import numpy as np
import rospy
from gymnasium import spaces
from gymnasium.envs.registration import register
from ur3_openai.ur3_env import UR3Env  # NOTE: Import your robot environment.
from ur_control.constants import GripperType, JOINT_ORDER

# Register the task environment as a gymnasium environment.
max_episode_steps = 1000
register(
    id="UR3TaskEnv-v0",
    entry_point="ur3_task_env:UR3TaskEnv",
    max_episode_steps=max_episode_steps,
)


class UR3TaskEnv(UR3Env):
    """Environment used to define the task the robot has to learn (i.e. observations
    you want to use, rewards etc).
    """

    def __init__(self):
        """Initializes a new Task environment."""
        # TODO: Implement the action space.
        self.num_joints = 6
        self.joint_lower_limit = [-np.pi,-np.pi/2,-np.pi/2,-np.pi,-np.pi,-np.pi]
        self.joint_upper_limit = [np.pi,np.pi/2,np.pi/2,np.pi,np.pi,np.pi]

        self.tran_upper_limit = [0.1,0.1,0.1,0.01,0.01,0.01] #  [x,y,z,roll,pitch,yaw]
        self.tran_lower_limit = [-0.1,-0.1,-0.1,-0.01,-0.01,-0.01] #  [x,y,z,roll,pitch,yaw]

        self.workspace_upper_limit = [0.5,0.7,0.6]#[x,y,z] ref on 'base_link'
        self.workspace_lower_limit = [-0.5,0.1,0.05]#y pointing to table, z pointing up

        # self.controllers_type = "joints_position"
        self.controllers_type = "ee_pose"
        if self.controllers_type == 'joints_position':
            self.action_space = self._get_joint_position_action_space()
        elif self.controllers_type == "ee_transform":
            self.action_space = self._get_ee_transform_action_space()
        elif self.controllers_type == "ee_pose":
            self.action_space = self._get_ee_pose_action_space()
        # number_actions = rospy.get_param("/my_robot_namespace/n_actions")
        # self.action_space = spaces.Discrete(number_actions)

        # TODO: Implement the observation space.
        
        self.observation_space = self._get_obs_space()
        # self.observation_space = spaces.Box(-high, high)

        # position goal
        self.goal = self.get_random_workspace_ee_position()
        self.goal_tolerance = 0.1
        self.cumulative_reward = 0
        # TODO: Retrieve required robot variables through the param server.

        # Initiate the Robot environment.
        super(UR3TaskEnv, self).__init__()

        

    ################################################
    # Task environment internal methods ############
    ################################################
    # NOTE: Here you can add additional helper methods that are used in the task env.
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
    
    def _get_ee_pose_action_space(self)-> spaces.Box:
        """
        Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        return spaces.Box(low=np.array(self.workspace_lower_limit), high=np.array(self.workspace_upper_limit), dtype=np.float32)
    
    def _get_obs_space(self)-> spaces.Box:
        """
        Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        high_obs = self.workspace_upper_limit + self.workspace_upper_limit + [1.0] # add goal and is_collided state
        low_obs = self.workspace_lower_limit + self.workspace_lower_limit + [0.0]
        return spaces.Box(low=np.array(low_obs), high=np.array(high_obs), dtype=np.float32)

    def get_random_workspace_ee_position(self):
        """
        Get a random position in the workspace.
        """
        return np.random.uniform(low=np.array(self.workspace_lower_limit), high=np.array(self.workspace_upper_limit), size=(1,3))

    def check_near_goal(self, ee_pose):
        return bool(np.linalg.norm(self.goal.reshape(1,3) - ee_pose.reshape(1,3))  < self.goal_tolerance)
    
    # def check_timeout(self,_):        
    #     try:
    #         if (rospy.Time.now() - self.js_timestamp).to_sec() > 1:
    #             rospy.logerr("No joint state received for more than 1 second")
    #             self.joint_states = None
    #         if (rospy.Time.now() - self.wrench_timestamp).to_sec() > 1:
    #             rospy.logerr("No wrench received for more than 1 second")
    #             self.wrench = None
    #     except Exception as e:
    #         rospy.logerr(f"Error in check_timeout: {str(e)}")
    
    ################################################
    # Overload Robot/Gazebo env virtual methods ####
    ################################################
    # NOTE: Methods that need to be implemented as they are called by the robot and
    # gazebo environments.
    def _set_init_gazebo_variables(self):
        """Initializes variables that need to be initialized at the start of the gazebo
        simulation.
        """
        # TODO: Implement logic that sets initial gazebo physics engine parameters.
        pass

    def _set_init_pose(self):
        """Sets the Robot in its initial pose."""
        # TODO: Implement logic that sets the robot to it's initial position.
        rospy.logwarn("Moving joint to initial position ...")
        q = self.joint_initial_positions
        self.move_joints(q, wait=True, target_time = 3.0)
        return True

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        # TODO: Reset variables that need to be reset at the start of each episode.
        self.is_ur3_collided = False
        self.goal = self.get_random_workspace_ee_position()
        self.cumulative_reward = 0

        return True

    def _get_obs(self):
        """Here we define what sensor data of our robots observations we have access to.

        Returns:
            numpy.ndarray: The observation data.
        """
        # TODO: Implement logic that retrieves the observation needed for the reward.
        ee_pose = self.arm.end_effector(rot_type='euler',tip_link='gripper_tip_link') #[x, y, z, roll, pitch, yaw] np.array
        is_collided = float(self.is_ur3_collided) # 1.0 or 0.0
        goal = self.goal

        observations = np.block([ee_pose[:3], goal,is_collided])  # [ee_x,ee_y,ee_z,goal_x,goal_y,goal_z,is_colided]

        return observations

    def _set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (numpy.ndarray): The action we want to apply.
        """
        # TODO: Implement logic that moves the robot based on a given action.
        if self.controllers_type == 'joints_position':
            q = action
            self.move_joints(q, wait=True, target_time = 3.0)
        elif self.controllers_type == "ee_transform":
            tran = action
            self.move_ee_relative(tran,wait=True,target_time= 2.0)
        elif self.controllers_type == "ee_pose":
            ee_pose = list(action) + [1.0,0.0,0.0,0.0]
            self.move_ee(ee_pose, wait=True, target_time= 4.0)
        

        return True

    def _is_done(self, observations):
        """Indicates whether or not the episode is done (the robot has fallen for
        example).

        Args:
            observations (numpy.ndarray): The observation vector.

        Returns:
            bool: Whether the episode was finished.
        """
        # TODO: Implement logic used to check whether a episode is done.
        is_near_goal = self.check_near_goal( observations[:3])
        if is_near_goal:
            rospy.loginfo("Reach Current Goal")

        is_colided = bool(observations[-1])
        if is_colided:
            rospy.logwarn("UR3 is collided")

        done = bool(is_near_goal or is_colided)
        return done

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.

        Args:
            observations (numpy.ndarray): The observation vector.
            done (bool): Whether the episode has finished.

        Returns:
            float: The step reward.
        """
        # TODO: Implement logic that is used to calculate the reward.
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
