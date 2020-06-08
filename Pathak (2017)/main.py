from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import tool
from agent import ICM_A2C_agent
import torch
from env_wrapper import FrameStack

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = FrameStack(env, 4)

obs = env.reset()

buffer = tool.buffer(20)
agent = ICM_A2C_agent(12)
agent.load()

for iteration in range(100000):
    if not (iteration) % 500:
        with torch.no_grad():
            obs = env.reset()
            done = False
            ret = 0
            while not done:
                env.render()
                act = agent.get_action(obs)
                nxt_obs, r_e, done, info = env.step(act)
                ret += r_e

        obs = env.reset()
        print(ret)
    
    buffer.reset()
    for step in range(20):
        act = agent.get_action(obs)
        nxt_obs, r_e, done, info = env.step(act)

        buffer.act.append(act)
        buffer.obs.append(obs.squeeze(0))
        buffer.nxt_obs.append(nxt_obs.squeeze(0))
        buffer.obs_feature.append(agent.icm.inverse.get_feature(obs))
        buffer.nxt_obs_feature.append(agent.icm.inverse.get_feature(nxt_obs))
        buffer.rwd.append(r_e)

        obs = nxt_obs

        if done:
            obs = env.reset()
            break

    loss_F, loss_I, loss_a2c = agent.train(buffer)
    print(f'epoch : {iteration} \t \
        L_F : {loss_F} \t \
        L_I : {loss_F} \t \
        L_a2c : {loss_a2c}')
