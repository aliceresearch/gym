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
    no_of_velocity_bins = 15
    no_of_position_bins = 19
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

    """
    0 position -1.2:0.6
    1 velocity -0.07:0.07
    """

    def act(self, observation, reward, step, done):
        print("observation {}",observation)
	action = 2
	if observation[1] < 0:
		action = 0
	elif observation[1] > 0:
		action = 2

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
    outdir = '/tmp/model-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QLearningAgent(env, 42)

    episode_count = 100
    reward = 0
    done = False
    total_steps = 100
    ob = env.reset()
    for step in range(total_steps):
        while True:
            action = agent.act(ob, reward, step, done)
	    print(action)
            ob, reward, done, _ = env.step(action)
            reward += reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
