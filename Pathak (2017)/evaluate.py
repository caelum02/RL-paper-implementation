from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import tool
from agent import ICM_A2C_agent
import torch
from env_wrapper import FrameStack
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = FrameStack(env, 4)

obs = env.reset()

buffer = tool.buffer(20)
agent = ICM_A2C_agent(12)
agent.load()

dist = []

with torch.no_grad():
    for i in range(5):
        cnt = 0
        maxX = 0
        done = False
        while not done:
            env.render()
            act = agent.get_action(obs)
            nxt_obs, r_e, done, info = env.step(act)
            obs = nxt_obs
            if maxX >= info['x_pos']+10:
                cnt += 1
            else:
                cnt=0
                maxX = info['x_pos']

            if done or cnt>300:
                obs = env.reset()
                dist.append(maxX)
                print(maxX)
                break

dist = np.array(dist)
print(dist.mean(), dist.std())