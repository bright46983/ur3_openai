#!/usr/bin/env python3
"""A small example script that shows how to use the `ros_gazebo_gym`_ package to train a
(SAC) agent to solve a `Panda`_ environment using `Stable Baselines3`_.

.. _ros_gazebo_gym: https://github.com/rickstaa/ros-gazebo-gym
"""
import os
import re

import gymnasium as gym
import numpy as np
import rospkg
import rospy
import torch
from ros_gazebo_gym.common.helpers import (
    list_2_human_text,
    to_pascal_case,
    to_snake_case,
)
from ros_gazebo_gym.core.helpers import ros_exit_gracefully
from ros_gazebo_gym.task_envs.task_envs_list import ENVS
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":  # noqa: C901
    rospy.init_node('ur3_sac') #,log_level=rospy.DEBUG

    env = gym.make(
        "UR3TaskEnv-v0"
    )

    # Initialize the logging system.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("ur3_openai")
    outdir = os.path.join(pkg_path, "training_results")
    last_time_steps = np.ndarray(0)

    # Load parameters from the ROS param server.
    # NOTE: Parameters are stored in a yaml files inside the config directory and
    # loaded at runtime by the launch file.
    alpha = rospy.get_param("/ur3_sac_params/alpha")
    gamma = rospy.get_param("/ur3_sac_params/gamma")
    n_episodes = rospy.get_param("/ur3_sac_params/n_episodes")
    n_steps = rospy.get_param("/ur3_sac_params/n_steps")
    inference = rospy.get_param("/ur3_sac_params/inference")
    inference_n_episodes = rospy.get_param(
        "/ur3_sac_params/inference_n_episodes"
    )
    total_timesteps = n_steps * n_episodes

    # Overwrite the max episode steps of the environment.
    env._max_episode_steps = n_steps

    # Convert goal gymnasium env to normal gymnasium env.
    env = gym.wrappers.FlattenObservation(env)
    
    
    # Check that the environment follows the Gym interface.
    check_env(env)

    # Initializes the algorithm that we are going to use for learning.
    torch.cuda.empty_cache()  # NOTE: Done to avoid CUDA out of memory error.
    model = SAC("MlpPolicy", env, verbose=1, gamma=gamma, learning_rate=alpha)

    # Train the algorithm.
    rospy.loginfo("Start training...")
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    rospy.loginfo("Training finished")

    # Safe policy.
    model_save_dir = os.path.join(outdir, "panda_baselines")
    model.save(str(model_save_dir))

    # Run inference.
    inference_ep_ret = []
    inference_ep_len = []
    if inference:
        rospy.logwarn("Starting inference...")
        obs, _ = env.reset()
        highest_reward = 0
        terminated, truncated, ep_ret, ep_len, n = False, False, 0, 0, 0
        rospy.logwarn("Episode: {}".format(n + 1))
        while n < inference_n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1

            if terminated or truncated:
                rospy.logdebug("Environment terminated or truncated. Resetting.")
                obs, _ = env.reset()
                inference_ep_ret.append(ep_ret)
                inference_ep_len.append(ep_len)
                terminated, truncated, ep_ret, ep_len, n = False, False, 0, 0, n + 1
                if n < inference_n_episodes:
                    rospy.logwarn("Episode: {}".format(n + 1))
        rospy.logwarn("Inference finished.")

        # Display inference diagnostics.
        mean_ep_ret = np.mean(inference_ep_ret)
        mean_ep_len = np.mean(inference_ep_len, dtype=np.int32)
        print("")
        rospy.logwarn("==Inference diagnostics==")
        rospy.logwarn(f"Mean EpRet: {mean_ep_ret}")
        rospy.logwarn(f"Mean EpLen: {mean_ep_len}")
        rospy.logwarn("=========================")

        env.close()
