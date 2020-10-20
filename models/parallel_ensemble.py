from torch import nn
import torch
from models.linearModel import BNN
import numpy as np
import time
import torch.multiprocessing as mp
# ctx = mp.get_context("spawn")
ctx = mp.get_context("fork")

class Ensemble(nn.Module):
    def __init__(self, state_dim, act_dim, pop_size, n_elites):
        super(Ensemble, self).__init__()
        self.pop_size = pop_size
        self.n_elites = n_elites
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.spawn_processes()
        self.elite_counter = np.zeros(self.pop_size)
        self.elite_idx = np.random.permutation(self.pop_size)[:self.n_elites]

    def spawn_processes(self):
        self.processes = []
        self.q_in = [ctx.Queue() for _ in range(self.pop_size)]
        self.q_out = [ctx.Queue() for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            p = ctx.Process(target=self.fork_controller, args=(self.q_in[i], self.q_out[i]))
            self.processes.append(p)
            p.start()
            self.q_in[i].put(i)
            assert self.q_out[i].get() == 'Ready'
            print('Process '+str(i)+' Started')

    def fork_controller(self, q_in, q_out):
        model = BNN(self.state_dim, self.act_dim)
        p_idx = q_in.get()
        q_out.put('Ready')
        while True:
            data, idx = q_in.get()
            if idx == p_idx:
                if len(data) == 1: # to
                    model.to(data[0])
                elif len(data) == 2: # pred
                    with torch.no_grad():
                        states, actions = data
                        pred = model(states, actions)
                        q_out.put(pred)
                elif len(data) == 3: # train
                    states, actions, targets = data
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
                    q_out.put((train_loss.item(), valid_loss.item()))
                elif data is None: # Exit
                    return 0
                else: # Other
                    print(len(data))
                    print('Process '+str(p_idx)+' exiting')
                    return 0

    def to(self, device):
        for i in range(self.pop_size):
            self.q_in[i].put(([device], i))
        return self

    def forward(self, states, actions):
        preds = []
        states, actions = states.detach(), actions.detach()
        start = time.time()
        for i in self.elite_idx:
            self.q_in[i].put(([states, actions], i))
        print(time.time()-start)
        for i in self.elite_idx:
            preds.append(self.q_out[i].get())
        return preds

    def train_step(self, states, actions, targets):
        assert len(states) == len(actions) == len(targets)
        valid_losses = []
        train_losses = []
        states, actions, targets = states.detach(), actions.detach(), targets.detach()
        for i in range(self.pop_size):
            self.q_in[i].put(([states, actions, targets], i))
        for i in range(self.pop_size):
            train_loss, valid_loss = self.q_out[i].get()
            valid_losses.append(valid_loss)
            train_losses.append(train_loss)

        self.elite_idx = np.argsort(valid_losses)[:self.n_elites]
        print('Train done')
        return np.mean(train_losses), np.mean(valid_losses)
