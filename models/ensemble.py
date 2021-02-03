from torch import nn
from models.parallel_ensemble import Ensemble as ParallelEnsemble
from models.seq_ensemble import Ensemble as SeqEnsemble
import torch
import numpy as np
from models.linearModel import BNN

class Ensemble(nn.Module):
    def __init__(self, state_dim, act_dim, pop_size=5, n_elites=5, parallel=False, Network=BNN):
        super(Ensemble, self).__init__()
        if parallel:
            self.ensemble = ParallelEnsemble(state_dim, act_dim, pop_size, n_elites, Network)
        else:
            self.ensemble = SeqEnsemble(state_dim, act_dim, pop_size, n_elites, Network)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        return self.ensemble.to(device)

    def forward(self, states, actions):
        return self.ensemble.forward(states, actions)

    def train_step(self, states, actions, targets):
        return self.ensemble.train_step(states, actions, targets)

    def train_set(self, states, actions, targets):
        return self.ensemble.train_set(states, actions, targets)

    def set_norm(self, mean,std):
        return self.ensemble.set_norm(mean, std)

    def sample(self, data, batch_size=256):
        P = np.random.permutation(len(data))[:batch_size]
        states, actions, targets = [], [], []
        for p in P:
            state, action, target = data[p]
            states.append(state)
            actions.append(action)
            targets.append(target)
        states, actions, targets = np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(targets, dtype=np.float32)
        states, actions, targets = torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(targets)
        states, actions, targets = states.to(self.device), actions.to(self.device), targets.to(self.device)
        return states, actions, targets
