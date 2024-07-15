import rospy
import gym
import numpy as np
from gym.utils import seeding
from openai_ros.gazebo_connection import GazeboConnection
from openai_ros.controllers_connection import ControllersConnection
from ur_control.arm import Arm
from ur_control.constants import GripperType


#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from openai_ros.msg import RLExperimentInfo
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelConfiguration, SetModelConfigurationRequest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import LoadControllerRequest, LoadController, UnloadController, UnloadControllerRequest, SwitchController, SwitchControllerRequest
import time

# https://github.com/openai/gym/blob/master/gym/core.py
class UR3GazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):

        # To reset Simulations
        rospy.logwarn("START init RobotGazeboEnv")

        self.model_name = 'robot'
        self.model_initial_pose = self.get_model_pose()

        self.reset_controls = reset_controls
        self.controllers_list = controllers_list
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        

        self.seed()
        
        

        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

        # We Unpause the simulation and reset the controllers if needed
        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        """
        self.gazebo.unpauseSim()
        if self.reset_controls:
            self.controllers_object.reset_controllers()

        rospy.logwarn("END init UR3GazeboEnv")

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.logwarn("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        rospy.logwarn("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self):
        rospy.logwarn("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logwarn("END Reseting RobotGazeboEnvironment")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logwarn("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        rospy.logwarn("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0


    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logwarn("RESET SIM START")
        if self.reset_controls :
            #### force reset ur3 model ########
            # self.gazebo.unpauseSim()
            # # self.stop_controllers() 
            # # self.unload_controllers()
            # # self.delete_model()
            # # self.spawn_urdf_model()
            
            # ###################################

            # self.gazebo.pauseSim()
            # time.sleep(2)

            # self.gazebo.resetSim()
            # self.set_model() # set gazebo joint state 
            # # time.sleep(2)

            # # rospy.sleep(3)
            # self.gazebo.unpauseSim()
            # rospy.logwarn("RESET SIM START")

            # # self.load_controllers()
            # # time.sleep(5)

            # self.controllers_object.reset_controllers()
            # time.sleep(20)

            # # self._check_all_systems_ready()
            # self.arm = Arm(gripper_type=GripperType.GENERIC)

            # self.gazebo.pauseSim()

            rospy.logwarn("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            ##### force reset ur3 model ######## 
            self.delete_model()
            self.spawn_urdf_model()
            ####################################
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logwarn("RESET SIM END")
        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()
    
    #################################
    # ur3 reset helper fucntion ######
    ##################################
    def js_cb(self,msg):
        self.js = msg

    def get_joint_initial_positions(self):
        '''
        subscibe to /joint_states until the first msg arrived --- save as initial state
        '''
        rospy.Subscriber('/joint_states', JointState, self.js_cb) 
        self.js = None
        # Wait for the message to arrive
        while not rospy.is_shutdown() and self.js is None:
            rospy.logwarn("Waiting for joint state to arrive")

        ang_list = list(self.js.position)
        name_list = self.js.name
        rospy.logwarn("Recived initial joint positions")

        return ang_list, name_list
    
    def set_model(self):
        rospy.wait_for_service('/gazebo/set_model_configuration')
        
        try:
            # Create a service proxy
            set_model_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
            
            # Model and configuration parameters
            req = SetModelConfigurationRequest()
            req.model_name = self.model_name  # Replace with your model name
            req.urdf_param_name = 'robot_description'  # Typically the parameter where the URDF is stored
            req.joint_names = self.joint_names
            req.joint_positions = self.joint_initial_positions

            print(req)
            

            # joint_positions = [0,0,0,0,0,0,0]
            # Call the service
            res = set_model_config(req)
            if res.success:
                rospy.loginfo("Set joint positions successfully.")
            else:
                rospy.logerr("Failed to set joint positions: %s", res.status_message)
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def get_model_pose(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            # Create a service proxy
            get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            
            # Call the service
            response = get_model_state_service(self.model_name, '')
            if response.success:
                rospy.loginfo("Model '%s' pose retrieved successfully", self.model_name)
                return response.pose
            else:
                rospy.logerr("Failed to get model state: %s", response.status_message)
                return None
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return None
    
    def delete_model(self):
        
        # Wait for the service to be available
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            # Create a service proxy
            delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            
            # Call the service
            delete_model_service(self.model_name)
            rospy.loginfo("Deleted model '%s' successfully", self.model_name)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def spawn_urdf_model(self):
        # Get the URDF XML from the ROS parameter server
        robot_description = rospy.get_param('/robot_description')

        # Define the initial pose for the model
        initial_pose = self.model_initial_pose
       
        # Name of the model to be spawned
        model_name = self.model_name

        # Create a service proxy to call the spawn URDF model service
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_urdf_model_prox = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            spawn_urdf_model_prox(model_name, robot_description, '', initial_pose, '')
            rospy.loginfo("Spawned URDF model successfully")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def load_controllers(self):
    
        rospy.wait_for_service('/controller_manager/load_controller')
        
        try:
            # Create a service proxy
            load_controller_service = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
            
            for name in self.controllers_list:
            # Create a request object
                request = LoadControllerRequest()
                request.name = name
            
                # Call the service
                response = load_controller_service(request)
                
                if response.ok:
                    rospy.loginfo(f"Controller '{name}' loaded successfully.")
                else:
                    rospy.logerr(f"Failed to load controller '{name}'.")
                    
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    
    def unload_controllers(self):
        rospy.wait_for_service('/controller_manager/unload_controller')
        try:
            unload_service = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
            for controller in self.controllers_list:
                response = unload_service(controller)
                if response.ok:
                    rospy.loginfo(f"Controller '{controller}' unloaded successfully.")
                else:
                    rospy.logerr(f"Failed to unload controller '{controller}'.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def stop_controllers(self):
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            switch_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            start_controllers = []
            stop_controllers = self.controllers_list
            strictness = 2  # STRICT
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = start_controllers
            switch_request_object.stop_controllers = stop_controllers
            switch_request_object.strictness = strictness

            response = switch_service(switch_request_object)
            if response.ok:
                rospy.loginfo(f"Controllers stopped successfully: {stop_controllers}")
            else:
                rospy.logerr(f"Failed to stop controllers: {stop_controllers}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")