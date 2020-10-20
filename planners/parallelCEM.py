import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CEM:
    def __init__(self, sim, act_space, rew_fn=None, nsteps=10):
        self.act_high = torch.from_numpy(act_space.high)
        self.act_low = torch.from_numpy(act_space.low)

        self.sim = sim
        if rew_fn is None:
            rew_fn = lambda obs: obs[0]
        self.rew_fn = rew_fn
        self.act_dim = sum(act_space.shape)
        self.nsteps=nsteps
        self.alpha = 0.1

    def plan_move(self, obs, init_mu=None, nsteps=None):
        if nsteps is None:
            nsteps = self.nsteps
        t, maxits, N, Ne, epsilon = 0, 20, 500, 50, 0.001
        if init_mu is None:
            mu = torch.zeros((nsteps, self.act_dim), device=device)
        else:
            mu = init_mu
        sigma = torch.ones((nsteps, self.act_dim), device=device)

        obs = torch.from_numpy(np.repeat(np.expand_dims(obs, 0), N, 0)).to(device).to(torch.float32)
        self.sim.init(obs)

        for i in range(nsteps):
            if not (mu[i] == 0).all():
                sigma[i] = 0.1
        while t < maxits and sigma.max() > epsilon:
            A = torch.normal(mu.unsqueeze(0).repeat(N, 1, 1), sigma.unsqueeze(0).repeat(N, 1, 1))
            A = A.clamp(-1, 1)
            state = self.sim.save()
            S = torch.stack([self.sim.sim_step(A[:,j]) for j in range(nsteps)])
            R = torch.stack([self.rew_fn(S[j].transpose(0,1)) for j in range(nsteps)])
            self.sim.load(state)
            R_sum = torch.sum(R, 0)
            A = A[torch.argsort(-R_sum)[:Ne]]
            new_mu, new_sigma = torch.mean(A, 0), torch.std(A, 0)

            mu = self.alpha * mu + (1-self.alpha) * new_mu
            sigma = self.alpha * sigma + (1 - self.alpha) * new_sigma
            t += 1

        # print('Pred R: '+str(R[0][torch.argmax(R_sum)].item()))
        return A[0]

if __name__ == '__main__':
    import gym
    import time
    import sys
    from sims.envSim import envSim
    nsteps = 10
    env = envSim(gym.make('Ant-v2').env)
    # env.render(mode='human')
    planner = CEM(env, env.action_space.shape[0])
    done = False
    obs = env.reset()
    ep_r = 0
    ep_l = 0
    n = 0
    action_seq = torch.zeros((nsteps, planner.act_dim), device=device)
    start = time.time()
    while True:
        next_seq = planner.plan_move(obs, action_seq, nsteps=nsteps)
        action = next_seq[0]
        action_seq[:-1] = next_seq[1:]
        new_obs, r, done, info = env.step(action)
        # print('Real R: '+str(r))
        # print('')
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds\t\t\r')
        ep_r += r
        ep_l += 1
        obs = new_obs
        # if done:
        if ep_l >= 1000:
            n += 1
            print('Episode '+str(n)+':')
            print('Ep R: '+str(ep_r))
            print('Ep L: '+str(ep_l))
            print('Time: '+str(time.time()-start))
            print('')
            done = False
            ep_r = 0
            ep_l = 0
            start = time.time()

            action_seq = np.zeros((nsteps, planner.act_dim))
            obs = env.reset()
    print('Finished')