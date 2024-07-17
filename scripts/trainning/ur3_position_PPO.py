#!/usr/bin/env python
import rospy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices from TensorFlow
import gym
from stable_baselines3 import PPO
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    rospy.init_node('ur3_ppo_training')
   
    env = StartOpenAI_ROS_Environment("UR3Position-v0")

    # Verify that the environment follows the Gym interface
    check_env(env)
    rospy.loginfo("Environment is compatible")
    # Define the PPO model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    rospy.loginfo("Starting training...")
    model.learn(total_timesteps=10000)
    rospy.loginfo("Training completed")

    # Save the model
    model.save("/tmp/ur3_ppo_model")

    rospy.loginfo("Model saved")

