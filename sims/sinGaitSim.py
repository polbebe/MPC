import gym
import numpy as np
from sims.sim import Sim
import pybullet as p

def act(obs, t, a, b, c):
    current_p = obs[:12]
    desired_p = np.zeros(12)
    v = a * np.sin(t * b) + c
    pos = [1, 10, 2, 11]
    neg = [4, 7, 5, 8]
    zero = [0, 3, 6, 9]
    desired_p[pos] = v
    desired_p[neg] = -v
    desired_p[zero] = 0

    delta_p = desired_p - current_p
    delta_p = np.clip(delta_p, -1, 1)
    return delta_p

class sinGaitSim(Sim, gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = 3
        self.observation_space = env.observation_space

    def reset(self):
        self.obs = self.env.reset()
        self.t = 0
        return np.concatenate([np.array([0]), self.obs])

    def step(self, action):
        self.obs, r, done, info =  self.env.step(action)
        self.t += 1
        return np.concatenate([np.array([r]), self.obs]), r, done, info

    def sim_step(self, params):
        R = 0
        for i in range(100):
            delta_p = act(self.obs, self.t, *params)
            new_obs, r, done, info =  self.step(delta_p)
            R += r
        new_obs = np.concatenate([np.array([R]), new_obs])
        return new_obs

    def save(self):
        p.saveBullet("state.bullet")
        return ["state.bullet", self.t, self.obs]

    def load(self, state):
        fileName, self.t, self.obs = state
        p.restoreState(fileName=fileName)

if __name__ == '__main__':
    import time
    env = gym.make('Ant-v2').env
    # env.render('human')
    env = sinGaitSim(env)
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
