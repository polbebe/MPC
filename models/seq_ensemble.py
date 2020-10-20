from torch import nn
import torch
from models.linearModel import BNN
import numpy as np

class Ensemble(nn.Module):
    def __init__(self, state_dim, act_dim, pop_size, n_elites):
        super(Ensemble, self).__init__()
        self.pop_size = pop_size
        self.n_elites = n_elites
        self.models = [BNN(state_dim, act_dim) for _ in range(self.pop_size)]
        self.elite_counter = np.zeros(self.pop_size)
        self.elite_idx = np.random.permutation(self.pop_size)[:self.n_elites]

    def to(self, device):
        for i in range(self.pop_size):
            self.models[i].to(device)
        return self

    def set_norm(self, mean,std):
        for i in range(self.pop_size):
            self.models[i].set_norm(mean, std)

    def forward(self, states, actions):
        # preds = [self.models[i](states, actions) for i in self.elite_idx]
        i = self.elite_idx[np.random.randint(0, self.n_elites)]
        # i = 0
        pred = self.models[i](states, actions)
        return pred

    def train_step(self, states, actions, targets):
        assert len(states) == len(actions) == len(targets)
        valid_losses = []
        train_losses = []
        for model in self.models:
            shuffle = np.random.permutation(len(states))
            train_len = int(0.875*len(states))
            train_idx = shuffle[:train_len]
            valid_idx = shuffle[train_len:]

            # This isn't a strict validation as what is chosen as validation here may be train another time
            # but it creates more diversity in the models by choosing only a subset and gives a little better validation
            train_states, train_actions, train_targets = states[train_idx], actions[train_idx], targets[train_idx]
            valid_states, valid_actions, valid_targets = states[valid_idx], actions[valid_idx], targets[valid_idx]
            train_loss = model.train_step(train_states, train_actions, train_targets)
            valid_loss = model.get_loss(valid_states, valid_actions, valid_targets)

            valid_losses.append(valid_loss.item())
            train_losses.append(train_loss)

        self.elite_idx = np.argsort(valid_losses)[:self.n_elites]

        return np.mean(train_losses), np.mean(valid_losses)
