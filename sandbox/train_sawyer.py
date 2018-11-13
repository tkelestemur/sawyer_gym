import os
import gym
import time
import numpy as np
import tensorflow as tf
from sawyer_env import SawyerEnv
from baselines import logger
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.tf_util import get_session
from baselines.ppo2 import ppo2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't output tensorflow warnings
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
get_session(config=config)


results_dir = "/home/tarik/sawyer_mujoco/results"
# env = SawyerEnv(reward_type='sparse')
env = gym.make("Reacher-v2")

logger_path = os.path.join(results_dir, str(0))
env = Monitor(env, logger_path, allow_early_resets=True)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env)


save_dir = "/home/tarik/sawyer_mujoco/data/"
model_name = "sawyer_reach_ppo2_1m"
network = 'mlp'

def train(save=False):

    logger.configure()
    model = ppo2.learn(network=network, env=env, total_timesteps=100000, nminibatches=32, lam=0.95,
                       gamma=0.99, noptepochs=10, log_interval=1, ent_coef=0)

    if save:
        model.save(save_dir + model_name)
        # env.save_running_average(save_dir)


def play():
    print(env.reset())


if __name__ == '__main__':
    train(True)
    # play()