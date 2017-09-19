import tempfile
import logging
import sys
import numpy as np
import gym
from gym import wrappers


""" Q-Learning Agent for experiments with the SDP Framework. Implementation of Q Learning from Watkins, Christopher John Cornish Hellaby. Learning from delayed rewards. Diss. University of Cambridge, 1989. Link (http://www.cs.rhul.ac.uk/~chrisw/thesis.html)

MountainCar Problem Space. Link (https://en.wikipedia.org/wiki/Mountain_Car)

State space
0 position [-1.2, 0.6] goal 0.5
1 velocity [-0.07, 0.07]

Action space
0	push left
1	no push
2	push right
"""
class QLearningAgent(object):
    seed = None
    actions = [0, 1, 2]
    epsilon = 0.01
    q = None
    previous_observation = None
    previous_action = None
    discount_factor = 0.1

    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        np.random.seed(seed)
        self.observation_space = env.observation_space
        # 3 actions and a positive/negative veloicty
        self.q = np.zeros(env.action_space.n * 2).reshape(3,2)

    def act(self, observation, reward, step):
        print("step {} reward {} observation {}".format(step, reward, observation))
        action = self.q.argmax(axis=0) + 1 if np.random.random() > self.epsilon else self.env.action_space.sample()
        if hasattr(action, "__iter__"):
            action = action[0]
        if self.previous_action is not None:
            learning_rate = 200 / (300 + step)
            current_q = self.q
            next_best_q = 1 # np.argmax(self.q[observation.item()])
            updated_q = (1 - learning_rate) * current_q + learning_rate * (reward + self.discount_factor * next_best_q)

        self.previous_observation = observation # for the next step
        self.previous_action = action # not in the observation?
        return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    outdir = tempfile.mkdtemp()
    env = wrappers.Monitor(env, directory=outdir, force=True)
    seed = 0
    env.seed(seed)
    agent = QLearningAgent(env, seed)

    total_reward = 0
    reward = 0
    done = False
    ob = env.reset()
    step = 0
    while True:
        action = agent.act(ob, reward, step)
        print("Action {}".format(action))
        ob, reward, done, _ = env.step(action)
        step = step + 1
        total_reward += reward
        if done:
            if ob[0] >= 0.5:
                print(("Solved in {} steps with a total reward of {}").format(step, total_reward))
            else:
                print(("Not solved in {} (max) steps with a total reward of {}").format(step, total_reward))
            break
    env.close()
