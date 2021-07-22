import pybullet as p
import time
import pybullet_data
import gym
import time
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import os

class PinkPantherEnv(gym.Env):
	def __init__(self, render=False):
		if render:
			self.physicsClient = p.connect(p.GUI)
			self.render = True
		else:
			self.render = False
		self.physicsClient = p.connect(p.DIRECT)
		self.mode = p.POSITION_CONTROL

		self.params = {
			'APS': 13, # Actions Per Second
			'maxForce': 1.667,
			'maxVel': 6.545,
			'gravity': -9.81,
			'act_noise': 0,
			'step': 3,
		}
		# TODO
		# setTimeStep should be uses here to modify the APS
		# Should keep a param of the joint states to be more in line with real life
		# need to figure out what the limits of the position control are

		self.stepper = 0

		self.ep_len = 0 # Ignore

		self.urdf = "envs/assets/PinkPanther_CML/urdf/pp_urdf_final.urdf"
		obs = self.reset()

		obs_high = np.ones_like(obs)
		self.observation_space = gym.spaces.Box(high=obs_high, low=-obs_high)
		act_high = np.ones(12)
		self.action_space = gym.spaces.Box(high=act_high, low=-act_high)

	def close(self):
		p.disconnect(self.physicsClient)
		# p.resetSimulation()

	def save_config(self, path):
		pass

	def load_config(self, path):
		pass

	def _render(self, mode='human', close=True):
		pass

	def get_obs(self):
		self.last_p = self.p
		self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
		self.p, self.q = np.array(self.p), np.array(self.q)
		self.v = self.p - self.last_p

		jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
		# jointVals = np.array([[joint[0], joint[1]] for joint in jointInfo]).flatten()
		# Pos only
		jointVals = np.array([joint[0] for joint in jointInfo]).flatten()
		obs = np.concatenate([jointVals, self.q, np.array([self.v[1], self.v[0]])])
		# print(self.v)
		return obs

	def act(self, action):
		action = action + np.random.normal(0, max(self.params['act_noise'], 0)) #??? Introduces noise ?
		n_sim_steps = int(240/self.params['APS'])
		for i in range(n_sim_steps):
			p.stepSimulation()
		for i in range(len(action)):
			pos, vel, forces, torque = p.getJointState(self.robotid, i)
			p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos+0.1*action[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel'])
			#p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel'])
		if self.render:
			time.sleep(1./self.params['APS'])

	def reset(self):
		p.resetSimulation()
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0,0,self.params['gravity'])
		planeId = p.loadURDF("envs/assets/PinkPanther_CML/plane/plane.urdf")
		robotStartPos = [0,0,0.2]
		robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
		self.robotid = p.loadURDF(self.urdf, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION)
		self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
		self.p, self.q = np.array(self.p), np.array(self.q)
		self.i = 0
		self.set()
		# print(self.p)
		return self.get_obs()

	def set(self):
		n_sim_steps = int(240/self.params['APS'])
		# Reset down
		for j in range(20):
		    for i in range(n_sim_steps):
		        p.stepSimulation()
		    pos = [0., 0.40854271, -0.9, 0., 0.4248062, -0.9, 0., 0.40854271, -0.9, 0., 0.4248062, -0.9]
		    for i in range(12):
		        p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel']/4)
		    if self.render:
		    	time.sleep(1./self.params['APS'])
		# Reset up
		for j in range(20):
		    for i in range(n_sim_steps):
		        p.stepSimulation()
		    vel = [1000, 1000, 1000, 1000, 1000, 1000, 700, 700, 700, 700, 700, 700]
		    for x in range(len(vel)):
		        vel[x] = vel[x]/250
		    for i in range(12):
		        p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=0, force=self.params['maxForce'], maxVelocity=self.params['maxVel']/vel[i])
		    if self.render:
		    	time.sleep(1./self.params['APS'])


	def step(self, action):
		if ((self.stepper != 0) and (self.stepper%self.params['step'] == 0)):
			self.act(action)
			obs = self.get_obs()
			self.i += 1
			self.stepper = 0
			done = self.i >= self.ep_len
			# r = 100*obs[0] - 50*np.abs(obs[1])
			r = 100*obs[-1] # positive x direction
			# r = 100*obs[-2] # positive y direction (aka turn left)
			# r = -100*obs[-2] # positive y direction (aka turn right)
			# r = 100*obs[-1] - 100*np.abs(obs[-2]) # positive x direction
			# print(self.p)
			return obs, r, done, {}
		else:
			self.stepper += 1
			obs = self.get_obs()
			done = self.i >= self.ep_len
			r = 100*obs[-1]
			return obs, r, done, {}

def get_action(state, steps):
	params = np.array([-0.62152756, -0.69011891, -0.93605901]) #1. Jul 13, 7pm

	#params = np.array([-0.84315363, -0.10856426,  0.39924414]) #1. Jul 13, 3pm
	#params = np.array([-0.62675961, -0.06858121,  0.32103562]) #2. Jul 13, 3pm
	#params = np.array([ 0.93501126, -0.74114365, 0.25697749]) #3. Jul 13, 3pm
	#params = np.array([-0.93977124, 0.04525157, -0.14929538]) #4. Jul 13, 3pm # best
	#params = np.array([-0.96662142, 0.03845074, -0.62658933]) #5. Jul 13, 3pm

	#params = np.array([-0.57717435, -0.05680442,  0.25591725]) #1. Jul 13, 12pm

	#params = np.array([0.85942766, 0.12127574, 0.29014669]) #1. Jul 13, 11am
	#params = np.array([0.65221345, 0.04103064, -0.34711618]) #2. Jul 13, 11am
	#params = np.array([0.94979543, 0.63588199, 0.09479404]) #3. Jul 13, 11am

	#params = np.array([-0.88687598, 0.09031087, -0.27228436]) # Jul 9, 5pm
	#params = np.array([0.95915886, 0.03045219, -0.11097666]) # Jul 9, 3pm
	#params = np.array([0.3, 0.1, 0])
	#params = np.array([0.9513956, -0.94153748, 0.03503142])
	#params = np.array([-0.57472189, -0.97479314, 0.04835059]) # with 10x actions
	#params = np.array([0.83972287, 0.79753211, 0.04102455]) # without 10x actions
	return act(state, steps, *params)

def act(obs, t, a, b, c):
	current_p = obs[:12]
	desired_p = np.zeros(12)
	v = a * np.sin(t * b) + c

	pos = [1, 10, 2, 11]
	neg = [4, 7, 5, 8]
	zero = [0, 3, 6, 9]
	desired_p[pos] = v
	desired_p[neg] = -v
	desired_p[zero] = 0

	delta_p = (desired_p - current_p)
	delta_p = np.clip(delta_p, -1, 1)
	return delta_p

def find_params(env, episodes):
	params = np.random.uniform(-1, 1, (episodes, 4))
	params[:,0] = 0
	for i in range(len(params)):
		print(i)
		params[i,0] = test_params(params[i,1:], env)
	p = np.argsort(params[:,0])[::-1]
	return params[p[0:5],1:]

def test_params(params, env, steps=100):
	ep_r = 0
	obs = env.reset()
	for t in range(steps):
		action = act(obs, t, *params)
		obs, r, done, info = env.step(action)
		ep_r += r
	return ep_r

'''
if __name__ == '__main__':

	env = PinkPantherEnv()
	#print(find_params(env, 500))
	
	obs = env.get_obs()

	for i in range(1000):
		action = get_action(obs, i)
		done = False
		while not done:
			obs, r, done, info = env.step(action)

	env.close()
'''