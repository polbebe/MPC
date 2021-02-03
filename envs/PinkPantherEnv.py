import pybullet as p
import time
import pybullet_data
import gym
import math as m
import numpy as np

class PinkPantherEnv(gym.Env):
    def __init__(self, render=False):
        if render:
            physicsClient = p.connect(p.GUI)
            self.render = True
        else:
            self.render = False
            physicsClient = p.connect(p.DIRECT)

        self.mode = p.POSITION_CONTROL
        self.APS=60
        self.maxForce = 4
        self.reset()

    def _render(self, mode='human', close=False):
        pass

    def get_obs(self):
        self.last_p = self.p
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.v = self.last_p - self.p
        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([[joint[0], joint[1]] for joint in jointInfo]).flatten()
        obs = np.concatenate([self.v, self.q, jointVals])
        return obs

    def act(self, action):
        n_sim_steps = int(240/self.APS)
        for i in range(n_sim_steps):
            p.stepSimulation()
        for i in range(len(action)):
            p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.maxForce)
        if self.render:
            time.sleep(1./self.APS)

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        planeId = p.loadURDF("plane/plane.urdf")
        robotStartPos = [0,0,0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.robotid = p.loadURDF("PinkPanther_CML/urdf/PinkPanther_CML.urdf", robotStartPos, robotStartOrientation)
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.i = 0
        return self.get_obs()

    def step(self, action):
        self.act(action)
        obs = self.get_obs()
        done = False
        r = -100*obs[0]
        return obs, r, done, {}

if __name__ == '__main__':
    def get_act(i):
        v = [0, 0, 0,
             -0.1, 0.2, 0,
             0.1, 0.2, 0,

             0, 0, 0,
             -0.1, -0.2, 0,
             0.1, -0.2, 0,

             0, 0, 0,
             -0.1, -0.2, 0,
             0.1, -0.2, 0,

             0, 0, 0,
             -0.1, 0.2, 0,
             0.1, 0.2, 0,

             0.1]
        armpit_lf = v[0] + v[1] * m.sin(i * v[36] + v[2])
        elbow_lf = v[3] + v[4] * m.sin(i * v[36] + v[5])
        knee_lf = v[6] + v[7] * m.sin(i * v[36] + v[8])
        armpit_rf = v[9] + v[10] * m.sin(i * v[36] + v[11])
        elbow_rf = v[12] + v[13] * m.sin(i * v[36] + v[14])
        knee_rf = v[15] + v[16] * m.sin(i * v[36] + v[17])
        armpit_lb = v[18] + v[19] * m.sin(i * v[36] + v[20])
        elbow_lb = v[21] + v[22] * m.sin(i * v[36] + v[23])
        knee_lb = v[24] + v[25] * m.sin(i * v[36] + v[26])
        armpit_rb = v[27] + v[28] * m.sin(i * v[36] + v[29])
        elbow_rb = v[30] + v[31] * m.sin(i * v[36] + v[32])
        knee_rb = v[33] + v[34] * m.sin(i * v[36] + v[35])
        act = np.array([armpit_lf, elbow_lf, knee_lf, armpit_rf, elbow_rf, knee_rf,
                        armpit_lb, elbow_lb, knee_lb, armpit_rb, elbow_rb, knee_rb])
        return act

    env = PinkPantherEnv(render=False)
    ep_r = 0
    start = time.time()
    for j in range(100):
        obs = env.reset()

        print(time.time()-start)
        for i in range(1000):
            action = get_act(i)
            obs, r, done, info = env.step(action)
            ep_r += r
        print(ep_r)
        print(time.time()-start)
        print('')
        ep_r = 0
