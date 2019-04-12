# -*- coding: utf-8 -*-

import gym
import time

# Create environment
env = gym.make('CartPole-v0')
env.reset()

# Loop through steps
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)
    
env.close()