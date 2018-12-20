import os
import tensorflow as tf
from gym.wrappers import TimeLimit
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs.sawyer_env import SawyerGraspEnv

env = SawyerGraspEnv(n_substeps=5)

env = TimeLimit(env, max_episode_steps=1000)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(PATH, 'results', 'baselines', 'ppo')

save_dir = "/home/tarik/sawyer_gym/results/baselines"
model_name = "/grasp/ppo"


def train(save=False):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    network = 'mlp'
    logger.configure()
    model = ppo2.learn(network=network, env=env, total_timesteps=1000000, nminibatches=32, lam=0.95,
                       gamma=0.99, noptepochs=10, log_interval=1, ent_coef=0)

    if save:
        model.save(save_dir + model_name)
        # env.save_running_average(save_dir)


if __name__ == '__main__':
    train(True)