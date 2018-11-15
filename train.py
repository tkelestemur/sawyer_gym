import os
from spinup import ppo, ddpg, sac
from spinup.utils.test_policy import load_policy, run_policy
from spinup.utils.mpi_tools import mpi_fork
import tensorflow as tf
from sawyer_reach import SawyerReachEnv

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = PATH + '/results/sawyer'
EXP_NAME = 'sawyer_reach'


def train():
    env_fn = lambda: SawyerReachEnv(n_substeps=10, reward_type='dense')

    ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
    logger_kwargs = dict(output_dir=SAVE_PATH, exp_name=EXP_NAME)

    mpi_fork(2)

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=1000,
        logger_kwargs=logger_kwargs, target_kl=0.01, max_ep_len=200)

    # ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)


def plot():
    pass


if __name__ == '__main__':
    train()