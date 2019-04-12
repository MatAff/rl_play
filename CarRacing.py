# -*- coding: utf-8 -*-

#from gym import envs
import gym
import time

env = gym.make('CarRacing-v0')
env.reset()

for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)
    
env.close()