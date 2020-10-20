
class Sim:
    def sim_step(self, action):
        raise NotImplementedError

    def init(self, obs):
        pass

    def save(self):
        raise NotImplementedError

    def load(self, state):
        raise NotImplementedError