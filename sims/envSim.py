import mujoco_py
import gym
import numpy as np
from sims.sim import Sim

class envSim(Sim, gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.xposbefore = self.env.get_body_com("torso")[0]
        return np.concatenate([np.array([0]), obs])

    def step(self, action):
        new_obs, r, done, info =  self.env.step(action)
        xposafter = self.env.get_body_com("torso")[0]
        r = (xposafter - self.xposbefore)/self.env.dt

        self.xposbefore = xposafter
        new_obs = np.concatenate([np.array([r]), new_obs])
        return new_obs, r, done, info

    def sim_step(self, action):
        new_obs, r, done, info =  self.step(action)
        new_obs = np.concatenate([np.array([r]), new_obs])
        return new_obs

    def save(self):
        return [self.env.sim.get_state(), self.xposbefore]

    def load(self, state):
        state, self.xposbefore = state
        self.env.sim.set_state(state)

if __name__ == '__main__':
    import time
    env = gym.make('Ant-v2').env
    # env.render('human')
    env = envSim(env)
    obs = env.reset()
    print('Init Obs: '+str(obs[:28]))

    action1 = env.action_space.sample()
    state = env.save()
    new_obs, r, done, info = env.step(action1)
    print('Real Obs 1: '+str(new_obs[:28]))
    n = 0
    start = time.time()
    ep_r = 0
    ep_l = 0
    for i in range(1000):
        action2 = env.action_space.sample()
        new_obs, r, done, info = env.step(action2)
        ep_r += 1
        ep_l += 1
        if done:
            n += 1
            print('Episode '+str(n)+':')
            print('Ep R: '+str(ep_r))
            print('Ep L: '+str(ep_l))
            print('Time: '+str(time.time()-start))
            ep_r = 0
            ep_l = 0
            env.reset()

        # time.sleep(0.01)

    env.load(state)
    replay_obs, replay_r, replay_done, replay_info = env.step(action1)
    print('Replay Obs 1: '+str(replay_obs[:28]))
    print('')
