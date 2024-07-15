#!/usr/bin/env python
import rospy
import gym
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


rospy.init_node('ur3_helloworld')

env = StartOpenAI_ROS_Environment("UR3Position-v0")
env.reset()

rate = rospy.Rate(30)

for n in range(5):
    for i in range(5):
        action = env.action_space.sample()
        rospy.logerr("---------------Episode:{}, Step:{}------------------".format(n,i))
        print(action)
        obs, reward, done, info = env.step(action) # take a random action
        if done:
            rospy.logerr("Episode:{} finish early------------------".format(n))

            break
        rospy.logerr("----------------------------------------------------")

        
    rospy.logerr("Finsihed Episode --> RESET")
    env.reset()
    rospy.logerr("RESET Completed")
rospy.spin()