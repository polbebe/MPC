import numpy as np

class CEM:
    def __init__(self, sim, act_space, rew_fn=None, nsteps=10):
        self.act_high = act_space.high
        self.act_low = act_space.low

        self.sim = sim
        if rew_fn is None:
            rew_fn = lambda obs: obs[0]
        self.rew_fn = rew_fn
        self.act_dim = sum(act_space.shape)
        self.nsteps=nsteps

    def plan_move(self, obs, init_mu=None, nsteps=None):
        self.sim.init(obs)

        if nsteps is None:
            nsteps = self.nsteps
        t, maxits, N, Ne, epsilon = 0, 64, 64, 8, 0.01
        if init_mu is None:
            mu = np.zeros((nsteps, self.act_dim))
        else:
            mu = init_mu
        sigma = np.ones((nsteps, self.act_dim))
        for i in range(nsteps):
            if not (mu[i] == 0).all():
                sigma[i] = 0.1
        while t < maxits and (sigma > epsilon).any():
            A = np.random.normal(mu, sigma, size=(N, nsteps, self.act_dim))
            S = []
            R = []
            for i in range(N):
                state = self.sim.save()
                s = np.stack([self.sim.sim_step(A[i][j]) for j in range(nsteps)])
                r = np.stack([self.rew_fn(s[j]) for j in range(nsteps)])
                # r = np.stack([self.rew_fn(self.sim.sim_step(X[i][j])) for j in range(nsteps)])
                self.sim.load(state)
                # assert self.sim.save() == state
                S.append(s)
                R.append(r)
            R, S = np.stack(R), np.stack(S)
            R_sum = np.sum(R, 1)
            A = A[np.argsort(-R_sum)[:Ne]]
            mu, sigma = np.mean(A, 0), np.std(A, 0)
            t += 1
        # print('Pred 1 Step Rew: '+str(R[np.argmax(R_sum)][0]))
        # print('Pred 2 Step Rew: '+str(R[np.argmax(R_sum)][1]))
        print(R[np.argmax(R_sum)][0])
        # print(np.max(R_sum))
        return A[0]
        # return mu

if __name__ == '__main__':
    import gym
    import time
    import sys
    from sims.envSim import envSim
    nsteps = 20
    env = envSim(gym.make('Ant-v2').env)
    # env.render(mode='human')
    planner = CEM(env, env.action_space.shape)
    done = False
    obs = env.reset()
    ep_r = 0
    ep_l = 0
    n = 0
    action_seq = np.zeros((nsteps, planner.act_dim))
    start = time.time()
    while True:
        next_seq = planner.plan_move(obs, action_seq, nsteps=nsteps)
        action = next_seq[0]
        action_seq[:-1] = next_seq[1:]
        new_obs1, r, done, info = env.step(action)
        sys.stdout.write('Step: '+str(ep_l)+' in '+str(round(time.time()-start,3))+' seconds\t\t\r')
        ep_r += r
        ep_l += 1
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