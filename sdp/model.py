import tempfile
import sys
import gym
from gym import wrappers


""" Model Agent for MountainCar which maximises the velocity by pushing 
in the direction of movement. The model agent can be optimised
by not pushing only part way up the left hill.

State space
0 position [-1.2, 0.6] goal 0.5
1 velocity [-0.07, 0.07]

Action space
0	push left
1	no push
2	push right
"""
class ModelAgent(object):

    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, step, done):
	if observation[1] < 0:
		action = 0
	elif observation[1] >= 0:
		action = 2

        return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    outdir = tempfile.mkdtemp()
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = ModelAgent(env)

    total_reward = 0
    reward = 0
    done = False
    ob = env.reset()
    step = 0
    while True:
    	action = agent.act(ob, reward, step, done)
    	ob, reward, done, _ = env.step(action)
	step = step + 1
	total_reward += reward
    	if done:
	    print(("Solved in {} steps with a total reward of {}").format(step, total_reward))
            break
    env.close()
