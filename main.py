from planners.parallelCEM import CEM
from models.linearModel import BNN
from models.ensemble import Ensemble
from sims.modelSim import modelSim
import gym
import torch
import numpy as np
import time
import sys

def train_model(env, model):
    batch_size = 256
    steps = 40000
    obs = env.reset()
    data = []
    for i in range(steps):
        action = env.action_space.sample()
        new_obs, r, done, info = env.step(action)
        # data.append((obs[:27], action, new_obs[:27]))
        data.append((obs, action, new_obs))
        obs = new_obs
        if len(data) > batch_size:
            states, actions, targets = model.sample(data, batch_size)
            in_data = torch.cat([states, actions], -1)
            batch_mean, batch_std = torch.mean(in_data, 0), torch.std(in_data, 0)
            # out_mean, out_std = torch.mean(targets, 0), torch.std(targets, 0)
            model.set_norm(batch_mean, batch_std)
            # model.set_norm(out_mean, out_std)
            loss = model.train_step(states, actions, targets)
            if (i+1)%100 == 0:
                print('Step '+str(i+1)+' Loss: '+str(loss))
                torch.save(model, 'linearModel.pt')
    all_states = torch.Tensor([step[0] for step in data])
    all_acts   = torch.Tensor([step[1] for step in data])
    all_inputs = torch.cat([all_states, all_acts], -1)

    all_mean, all_std = torch.mean(all_inputs, 0), torch.std(all_inputs, 0)
    model.set_norm(all_mean, all_std)
    return model

def plan_episode(env, planner):
    ep_r = 0
    ep_l = 0
    done = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_seq = torch.zeros((planner.nsteps, planner.act_dim), device=device)

    obs = env.reset()

    start = time.time()
    while not done:
        obs = obs[:28]
        next_seq = planner.plan_move(obs, action_seq, nsteps=planner.nsteps)
        action = next_seq[0]
        # action = env.action_space.sample()
        # new_obs, r, done, info = env.step(action)
        # action_seq[:-1] = next_seq[1:]
        new_obs, r, done, info = env.step(action.cpu().numpy())
        obs = new_obs
        # print('Real R: '+str(r))
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds\tRew So Far: '+str(round(ep_r,2))+'                         \r')
        ep_r += r
        ep_l += 1

        done = ep_l >= 1000

    print('Ep R: '+str(ep_r))
    print('Ep L: '+str(ep_l))
    print('Time: '+str(time.time()-start))
    print('')

# Results:
# Null (aka all 0 action vectors): 0
# Random: -350
# Learned @10K w/ 20 step lookahead: 275
# Learned @10K w/ 20 step lookahead and act seq: -1300
# Perfect w/ 20 step lookahead: 7000

# Learned @ 25K w/ 10 step lookahead halfcheetah 350
# Learned @ 25K w/ 20 step lookahead halfcheetah 740
# Learned @ 25K w/ 30 step lookahead halfcheetah 740

# Ant @ 100K w/ 20 steps and stochastic action 788

class rewWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        inf = np.array([np.inf])
        high = np.concatenate([inf, env.observation_space.high])
        low = np.concatenate([-inf, env.observation_space.low])
        self.observation_space = gym.spaces.Box(high=high, low=low)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([np.array([0]), obs])

    def step(self, action):
        new_obs, r, done, info =  self.env.step(action)
        new_obs = np.concatenate([np.array([r]), new_obs])
        return new_obs, r, done, info

if __name__ == '__main__':
    # env = gym.make('Ant-v2').env
    env = gym.make('HalfCheetah-v2').env
    env = rewWrapper(env)

    # state_dim, act_dim = 28, env.action_space.shape[0]
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    # model = Ensemble(state_dim, act_dim)
    # model.to(model.device)
    # train_model(env, model)
    # torch.save(model, 'halfcheetah_norm_ensemble.pt')
    # torch.save(model, 'halfcheetah_nonorm_ensemble.pt')
    model = torch.load('halfcheetah_norm_ensemble.pt')
    # model = torch.load('ensemble.pt')
    # model = torch.load('ant_100K_norm_ensemble.pt')
    print('Model Prepared')
    nsteps = 20

    planner = CEM(modelSim(model), env.action_space, nsteps=nsteps)
    print('Planner initialized')
    plan_episode(env, planner)
    print('')
    print('Finished')