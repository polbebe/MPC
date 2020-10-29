from torch import nn
from torch import optim
import torch
import numpy as np

class BNN(nn.Module):
    def __init__(self, state_dim, act_dim, n_hid=200, lr=1e-3):
        super(BNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim+act_dim, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, state_dim*2)
                                            )
        self.softplus = nn.Softplus()
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_mean = None

    def set_norm(self, mean,std):
        self.norm_mean = mean.to(self.device).float()
        self.norm_std = std.to(self.device).float()
        self.norm_std[self.norm_std < 1e-12] = 1.0

    def forward(self, states, actions, train=False):
        if not torch.is_tensor(states):
            states = torch.from_numpy(states).to(self.device).to(torch.float32)
        if not torch.is_tensor(actions):
            actions = torch.from_numpy(actions).to(self.device).to(torch.float32)
        x = torch.cat([states, actions], dim=-1)

        # Input Normalization
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std

        out = self.model(x)
        mean, logvar = torch.chunk(out, 2, -1)
        mean = mean + states
        max_logvar = torch.ones_like(logvar)
        min_logvar = -torch.ones_like(logvar)
        logvar = max_logvar - self.softplus(max_logvar - logvar)
        logvar = min_logvar + self.softplus(logvar - min_logvar)
        if train:
            return mean, logvar
        else:
            # mean = mean + states
            var = torch.exp(logvar)
            return mean + torch.randn_like(mean, device=mean.device) * var.sqrt()
            # return mean

    def loss(self, pred, target, include_var=True):
        if include_var:
            mean, log_var = pred
            inv_var = torch.exp(-log_var)
            mse_loss = torch.mean(((mean - target) ** 2) * inv_var)
            var_loss = torch.mean(log_var)
            l = mse_loss + var_loss
        else:
            mean = pred
            mse_loss = torch.mean((mean - target))
            l = mse_loss
        return l

    def get_loss(self, states, actions, targets):
        # targets = targets - states

        preds = self.forward(states, actions, train=True)
        loss = self.loss(preds, targets)
        return loss

    def train_step(self, states, actions, targets):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(states, actions, targets)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def train_set(self, all_states, all_acts, all_targets):
        losses = []
        i = 0
        self.model.train()
        self.optimizer.zero_grad()
        p = np.random.permutation(len(data))
        while i < len(data):
            idx = p[i:i+batch_size]
            states, actions, targets = all_states[idx], all_acts[idx], all_targets[idx]
            states = states.to(model.device)
            actions = actions.to(model.device)
            targets = targets.to(model.device)
            loss = self.get_loss(states, actions, targets)
            loss.backward()
            losses.append(loss.item())
            i += batch_size
        self.optimizer.step()
        self.model.eval()
        return np.mean(losses)

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

if __name__ == '__main__':
    import gym

    batch_size = 256
    steps = 10000
    env = gym.make('Ant-v2').env
    obs = env.reset()
    data = []
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    model = BNN(state_dim, act_dim)
    model = model.to(model.device)
    for i in range(steps):
        action = env.action_space.sample()
        new_obs, r, done, info = env.step(action)
        data.append((obs, action, new_obs))
        obs = new_obs
        if len(data) > batch_size:
            states, actions, targets = model.sample(data, batch_size)
            loss = model.train_step(states, actions, targets)
            if (i+1)%100 == 0:
                print('Step '+str(i+1)+' Loss: '+str(loss))
                torch.save(model, 'linearModel.pt')