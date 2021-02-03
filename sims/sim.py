import torch
class Sim:
    def sim_step(self, action):
        raise NotImplementedError

    def init(self, obs):
        pass

    def save(self):
        raise NotImplementedError

    def load(self, state):
        raise NotImplementedError

class modelSim(Sim):
    def __init__(self, model):
        self.model = model
        self.state = None

    def init(self, obs):
        self.state = obs

    def sim_step(self, action):
        new_obs = self.model(self.state, action)
        new_obs = new_obs.detach()
        if not torch.is_tensor(action):
            new_obs = new_obs.cpu().numpy()
        self.state = new_obs
        return new_obs

    def save(self):
        return self.state

    def load(self, state):
        self.state = state