import numpy as np

class CEM:
    def __init__(self, sim, act_space, rew_fn=None, nsteps=10):
        try:
            self.act_high = act_space.high
            self.act_low = act_space.low
        except:
            self.act_high, self.act_low = None, None

        self.sim = sim
        if rew_fn is None:
            rew_fn = lambda obs: obs[0]
        self.rew_fn = rew_fn
        self.rew_fn = rew_fn
        try:
            self.act_dim = sum(act_space.shape)
        except:
            self.act_dim = act_space
        self.nsteps=nsteps
        self.alpha = 0.1

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
        best_act = None
        best_r = None
        r_avg = None
        while t < maxits and (sigma > epsilon).any():
            A = np.random.normal(mu, sigma, size=(N, nsteps, self.act_dim))
            S = []
            R = []
            for i in range(N):
                state = self.sim.save()
                s = np.stack([self.sim.sim_step(A[i][j]) for j in range(nsteps)])
                r = np.stack([self.rew_fn(s[j]) for j in range(nsteps)])
                self.sim.load(state)
                S.append(s)
                R.append(r)
            R, S = np.stack(R), np.stack(S)
            R_sum = np.sum(R, 1)
            A = A[np.argsort(-R_sum)[:Ne]]
            new_mu, new_sigma = np.mean(A, 0), np.std(A, 0)
            if r_avg is None or np.mean(R_sum[np.argsort(-R_sum)[:Ne]]) > r_avg:
                r_avg = np.mean(R_sum[np.argsort(-R_sum)[:Ne]])
                mu = self.alpha * mu + (1-self.alpha) * new_mu
                sigma = self.alpha * sigma + (1 - self.alpha) * new_sigma
            t += 1
            if best_r is None or np.max(R_sum) > best_r:
                best_r = np.max(R_sum)
                best_act = A[0]
            print(t, r_avg, np.max(R_sum), np.max(sigma))
        print(best_r, best_act)
        return best_act

if __name__ == '__main__':
    import time
    import sys
    from sims.sinGaitSim import sinGaitSim, act
    from envs.pinkpanther import PinkPantherEnv
    nsteps = 1
    env = PinkPantherEnv(render=True)
    env = sinGaitSim(env)
    # env.render(mode='human')
    planner = CEM(env, env.action_space)
    done = False
    obs = env.reset()
    ep_r = 0
    ep_l = 0
    n = 0
    action_seq = np.zeros((nsteps, planner.act_dim))
    start = time.time()
    params = planner.plan_move(obs, action_seq, nsteps=nsteps)
    print(params)
    env = PinkPantherEnv(render=True)
    while True:
        R = 0
        obs = env.reset()
        for t in range(1000):
            action = act(obs, t, *params)
            obs, r, done, info = env.step(action)
            time.sleep(0.1)
            R += r
        print(R)
        time.sleep(5)