#!/usr/bin/env python
from __future__ import print_function
from atari_helperfunc_tutorial import *
from ring_buf import *

import sys, gym, time

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

actions = np.arange(0,4)

is_done = False
while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()
    time.sleep(0.1)
