from planners.parallelCEM import CEM
from models.linearModel import BNN
from models.ensemble import Ensemble
from sims.modelSim import modelSim
import gym
import torch
import numpy as np
import time
import sys

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
            done = steps >= 1000
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

def train_model(env, model):
    batch_size = 256
    steps = 10000
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

def train_on_policy(env, model, planner):
    steps = 100000
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
        if ep_l == 1000:
            print('')
            for _ in range(8):
                loss = train_epoch(model, data, batch_size=32)
            print('Step '+str(i+1)+' Loss: '+str(loss))
            # torch.save(model, 'linearModel.pt')
            print('Total Steps: '+str(i))
            print('Ep R: ' + str(ep_r))
            print('Ep L: ' + str(ep_l))
            print('Time: ' + str(time.time() - start))
            print('')
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
    ep_r = 0
    ep_l = 0
    done = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_mu = torch.zeros((planner.nsteps, planner.act_dim), device=device)
    action_sigma = torch.ones((planner.nsteps, planner.act_dim), device=device)

    obs = env.reset()
    obs_hist = [obs]
    start = time.time()
    while not done:
        next_mu, next_sigma = planner.plan_move(obs, action_mu, action_sigma, nsteps=planner.nsteps)
        action = next_mu[0].cpu().numpy()

        action_mu[:-1] = next_mu[1:]
        action_sigma[:-1] = next_sigma[1:]

        # action = env.action_space.sample()
        new_obs, r, done, info = env.step(action)
        obs = new_obs
        # print('Real R: '+str(r))
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds\tRew So Far: '+str(round(ep_r,2))+'                         \r')
        ep_r += r
        ep_l += 1

        done = ep_l >= 1000
        obs_hist.append(obs)

    print('Ep R: '+str(ep_r))
    print('Ep L: '+str(ep_l))
    print('Time: '+str(time.time()-start))
    print('')
    return obs_hist

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
    nsteps = 20
    from envs.halfcheetah import HalfCheetahEnv
    from envs.pinkpanther import PinkPantherEnv
    # env = gym.make('Ant-v2').env
    # env = gym.make('HalfCheetah-v1').env
    # env = PinkPantherEnv(render=True)
    env = gym.make('HalfCheetah-v2').env
    # env = gym.make('Pusher-v2').env
    env = rewWrapper(env)
    # env = HalfCheetahEnv()

    # run_rand(env)

    # state_dim, act_dim = 28, env.action_space.shape[0]
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    model = Ensemble(state_dim, act_dim)
    model.to(model.device)

    # train_model(env, model)
    # torch.save(model, 'pinkpanther_ensemble.pt')
    # model = torch.load('pinkpanther_ensemble.pt')

    planner = CEM(modelSim(model), env.action_space, nsteps=nsteps)
    train_on_policy(env, model, planner)

    # torch.save(model, 'halfcheetah_100K_on_policy_norm_ensemble.pt')
    torch.save(model, 'halfcheetah_norm_ensemble.pt')
    # model = torch.load('halfcheetah_norm_ensemble.pt')
    # model = torch.load('ensemble.pt')
    # model = torch.load('ant_100K_norm_ensemble.pt')
    # model = torch.load('halfcheetah_100K_on_policy_norm_ensemble.pt')
    # model = torch.load('reacher_norm_ensemble.pt')
    # print('Model Prepared')
    # print('Planner initialized')
    pos_paths = []
    for i in range(10):
        obs_hist = plan_episode(env, planner)
        pos_path = []
        pos = np.zeros(2)
        for obs in obs_hist:
            pos += obs[1:3]
            pos_path.append(pos.copy())
        pos_paths.append(np.array(pos_path))
    print('')
    print('Finished')
    pos_paths = np.array(pos_paths)
    means = np.mean(pos_paths, 0)
    stds = np.std(pos_paths, 0)
    import matplotlib.pyplot as plt
    plt.title('MPC Action')
    plt.axis('equal')
    plt.plot(means[:,0], means[:,1])
    plt.plot(0, 0, 'rx')
    plt.fill_between(means[:,0], means[:,1]-stds[:,1], means[:,1]+stds[:,1], alpha=0.5)
    plt.show()