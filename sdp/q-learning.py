import argparse
import logging
import sys
# import random
import numpy as np
import gym
from gym import wrappers


""" Q-Learning Agent for experiments with the SDP Framework. Implementation of Q Learning from Watkins, Christopher John Cornish Hellaby. 
Learning from delayed rewards. Diss. University of Cambridge, 1989.

Link (http://www.cs.rhul.ac.uk/~chrisw/thesis.html)

"""
class QLearningAgent(object):
    seed = 0
    accumulated_reward = 0
    no_of_velocity_bins = 2
    no_of_actions = 3
    epsilon = 0.01
    q = None
    previous_observation = None
    previous_action = None

    def __init__(self, env, seed):
        self.env = env
        self.action_n = env.action_space.n
        self.observation_space = env.observation_space
        np.random.seed(seed)
        init_q = np.zeros(self.no_of_velocity_bins * self.no_of_actions, dtype=np.float)
        self.q = init_q.reshape(self.no_of_velocity_bins, self.no_of_actions)

    """
    0 position -1.2:0.6
    1 velocity -0.07:0.07
    """
    def bin_index(self, value):
        if value > 0:
            return 0
        else:
            return 1

    def act(self, observation, reward, step, done):
        print("observation {}",observation[0])
        # print("{}", )
        v_bin = self.bin_index(observation[1])
        print("v_bin {} p_bin {}", v_bin)
        action_q = self.q[v_bin]
        print(action_q)
        action = action_q.argmax(axis=0) if np.random.random() > self.epsilon else self.env.action_space.sample()
        # action = np.amax(q[v_bin, p_bin]) if np.random.random() > self.epsilon else self.env.action_space.sample()

        # was there a  previous action to update the q
        if self.previous_action is not None:
            learning_rate = 200 / (300 / step)
            current_q = self.q
            next_best_q = np.argmax(self.q[observation.item()])
            updated_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * next_best_q)
        # double learningRate = (200.0 / (300.0 + timestep));
        # double currentQ = QValues[state.center + offset][action.getValue() + offset];
        # double QForNextState = getBestQForState(nextState);
        # double updatedQValue =
        #         (1.0 - learningRate) * currentQ + learningRate * (reward + discountFactor * QForNextState);
        # QValues[state.center + offset][action.getValue() + offset] = updatedQValue;

        previous_observation = observation # for the next step
        previous_action = action # not in the observation?
        
        return action

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/q-learning-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(42)
    agent = QLearningAgent(env, 42)

    episode_count = 100
    reward = 0
    done = False
    total_steps = 100
    ob = env.reset()
    for step in range(total_steps):
        while True:
            action = agent.act(ob, reward, step, done)
            ob, reward, done, _ = env.step(action)
            reward += reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
