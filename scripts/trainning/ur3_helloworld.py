#!/usr/bin/env python
import rospy
import gymnasium as gym
from gymnasium.envs.registration import register
# Register the task environment as a gymnasium environment.
max_episode_steps = 1000
register(
    id="UR3TaskEnv-v0",
    entry_point="ur3_openai.ur3_task_env:UR3TaskEnv",
    max_episode_steps=max_episode_steps,
)

rospy.init_node('ur3_helloworld',log_level=rospy.DEBUG)

env = gym.make(
        "UR3TaskEnv-v0"
    )

env.reset()

rate = rospy.Rate(30)

for n in range(5):
    for i in range(5):
        action = env.action_space.sample()
        rospy.logerr("---------------Episode:{}, Step:{}------------------".format(n,i))
        print(action)
        obs, reward, done,_, info = env.step(action) # take a random action
        if done:
            rospy.logerr("Episode:{} finish early------------------".format(n))

            break
        rospy.logerr("----------------------------------------------------")

        
    rospy.logerr("Finsihed Episode --> RESET")
    env.reset()
    rospy.logerr("RESET Completed")
rospy.spin()