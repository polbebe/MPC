from planners.parallelCEM import CEM
from models.linearModel import BNN
from models.ensemble import Ensemble
from sims.modelSim import modelSim
import gym
import torch
import numpy as np
import time
import sys

max_ep_len = 100

def run_rand(env):
    import matplotlib.pyplot as plt
    episodes = 100
    rs = np.zeros(episodes)
    for i in range(episodes):
        obs, done, ep_r, steps = env.reset(), False, 0.0, 0.0
        all_pos = []
        pos = np.zeros(2)
        while not done:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            pos += obs[:2]
            all_pos.append(pos.copy())
            steps += 1
            done = steps >= max_ep_len
            ep_r += r
        rs[i] = ep_r
        # print(ep_r)
        # all_pos = 100*np.array(all_pos)
        # plt.plot(all_pos[:,0], all_pos[:,1])
        # plt.axis('equal')
        # plt.show()
        sys.stdout.write('Episode '+str(i)+'/'+str(episodes)+'                   \r')
    print('')
    print('Rand Mean: '+str(np.mean(rs)))
    print('Rand Std:  '+str(np.std(rs)))

def train_model(steps, env, model, data=[], batch_size=64):
    # batch_size = min(int(steps/2), batch_size)
    obs = env.reset()
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
            # if (i+1)%(steps/10) == 0:
            #     print('Step '+str(i+1)+' Loss: '+str(loss))
            #     torch.save(model, 'linearModel.pt')
    # all_states = torch.Tensor([step[0] for step in data])
    # all_acts   = torch.Tensor([step[1] for step in data])
    # all_inputs = torch.cat([all_states, all_acts], -1)
    # all_mean, all_std = torch.mean(all_inputs, 0), torch.std(all_inputs, 0)
    # model.set_norm(all_mean, all_std)
    return model, loss, data

def train_epoch(model, data, batch_size=32):
    all_states = torch.Tensor([step[0] for step in data])
    all_acts   = torch.Tensor([step[1] for step in data])
    all_targets= torch.Tensor([step[2] for step in data])
    all_inputs = torch.cat([all_states, all_acts], -1)
    all_mean, all_std = torch.mean(all_inputs, 0), torch.std(all_inputs, 0)
    model.set_norm(all_mean, all_std)
    i = 0
    losses = []
    p = np.random.permutation(len(data))
    while i < len(data):
        idx = p[i:i+batch_size]
        states, actions, targets = all_states[idx], all_acts[idx], all_targets[idx]
        states = states.to(model.device)
        actions = actions.to(model.device)
        targets = targets.to(model.device)
        loss = model.train_step(states, actions, targets)
        losses.append(loss[0])
        i += batch_size
    return np.mean(losses)

def train_on_policy(env, model, planner, steps, update_every=100):
    obs = env.reset()
    data = []
    ep_l = 0
    ep_r = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    action_mu = torch.zeros((planner.nsteps, planner.act_dim), device=device)
    action_sigma = torch.ones((planner.nsteps, planner.act_dim), device=device)*0.25
    for i in range(steps):
        # action = env.action_space.sample()
        if i > 1000:
            next_mu, next_sigma = planner.plan_move(obs, action_mu, action_sigma, nsteps=planner.nsteps)
            action = next_mu[0].cpu().numpy()

            action_mu[:-1] = next_mu[1:]
            action_sigma[:-1] = next_sigma[1:]
        else:
            action = env.action_space.sample()

        new_obs, r, done, info = env.step(action)
        ep_r += r
        ep_l += 1
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds    Rew So Far: '+str(round(ep_r,2))+'                         \r')
        # data.append((obs[:27], action, new_obs[:27]))
        data.append((obs, action, new_obs))
        obs = new_obs
        if i%update_every==update_every-1:
            for _ in range(8):
                loss = train_epoch(model, data, batch_size=32)
            # print('Step '+str(i+1)+' Loss: '+str(loss))
        if ep_l == max_ep_len:
            # print('')
            # # torch.save(model, 'linearModel.pt')
            # print('Total Steps: '+str(i))
            # print('Ep R: ' + str(ep_r))
            # print('Ep L: ' + str(ep_l))
            # print('Time: ' + str(time.time() - start))
            # print('')
            ep_r = 0
            ep_l = 0
            action_mu = torch.zeros((planner.nsteps, planner.act_dim), device=device)
            action_sigma = torch.ones((planner.nsteps, planner.act_dim), device=device)*0.25
            obs = env.reset()
    all_states = torch.Tensor([step[0] for step in data])
    all_acts   = torch.Tensor([step[1] for step in data])
    all_inputs = torch.cat([all_states, all_acts], -1)

    all_mean, all_std = torch.mean(all_inputs, 0), torch.std(all_inputs, 0)
    model.set_norm(all_mean, all_std)
    return model

def plan_episode(env, planner):
    ep_r, ep_l = 0, 0
    done = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_mu = torch.zeros((planner.nsteps, planner.act_dim), device=device)
    action_sigma = torch.ones((planner.nsteps, planner.act_dim), device=device)

    obs = env.reset()
    obs_hist = [obs]
    start = time.time()

    pred_ep_r = 0
    pred_rs = []
    real_rs = []
    while not done:
        next_mu, next_sigma, pred_r = planner.plan_move(obs, action_mu, action_sigma, nsteps=planner.nsteps)
        pred_ep_r += pred_r
        action = next_mu[0].cpu().numpy()

        action_mu[:-1] = next_mu[1:]
        action_sigma[:-1] = next_sigma[1:]

        # action = env.action_space.sample()
        new_obs, r, done, info = env.step(action)
        obs = new_obs
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds\tRew So Far: '+str(round(ep_r,2))+'                         \r')
        ep_r += r
        ep_l += 1

        done = ep_l >= max_ep_len
        pred_rs.append(pred_r)
        real_rs.append(r)
        obs_hist.append(obs)

    # print('Ep R: '+str(ep_r))
    # print('Pred Ep R: '+str(pred_ep_r))
    # print('Ep L: '+str(ep_l))
    # print('Time: '+str(time.time()-start))
    # print('')
    # return obs_hist, ep_r, pred_ep_r
    return obs_hist, real_rs, pred_rs

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
    nsteps = 1
    from envs.pinkpanther import PinkPantherEnv
    import pickle as pkl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PinkPantherEnv(render=False)
    env = rewWrapper(env)
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # steps = 100
    # train_steps = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    # train_steps = np.concatenate([np.arange(10)*100, np.arange(1,51)*1000])
    train_steps = np.arange(201)*100
    n_trials, n_runs = 10, 5
    # results = np.zeros((len(train_steps), 1+2*n_trials))
    results, losses = dict(), dict()
    # results = np.load('MPC_results.npy')
    pos_paths = []
    print('Starting')
    # for steps in train_steps:
    models = [Ensemble(state_dim, act_dim) for _ in range(n_runs)]
    datas = [[] for i in range(n_runs)]
    for i in range(len(train_steps)):
        start = time.time()
        results[train_steps[i]] = []
        losses[train_steps[i]] = []
        for j in range(n_runs):
            pred_rs, real_rs = [], []
            # models = [Ensemble(state_dim, act_dim) for _ in range(n_runs)]
            # model = Ensemble(state_dim, act_dim)
            # model.to(model.device)
            if train_steps[i] > 0:
                models[j], loss, datas[j] = train_model(100, env, models[j], datas[j])
                losses[train_steps[i]].append(loss)
                pkl.dump(losses, open('losses.pkl', 'wb+'))
            planner = CEM(modelSim(models[j]), env.action_space, nsteps=nsteps)
            print(str(train_steps[i]) +' Model trained in '+str(round(time.time()-start,3))+'s                                              ')
            for _ in range(n_trials):
                obs_hist, ep_r, pred_ep_r = plan_episode(env, planner)
                real_rs.extend(ep_r)
                pred_rs.extend(pred_ep_r)
            results[train_steps[i]].append((real_rs, pred_rs))
            pkl.dump(results, open('MPC_results.pkl', 'wb+'))
        # print([len(d) for d in datas])
        assert train_steps[i] == np.mean([len(d) for d in datas])
        print(str(train_steps[i]) +' Finished in '+str(round(time.time()-start,3))+'s')
    print('')
    print('Finished')
