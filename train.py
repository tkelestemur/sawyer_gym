import os
from spinup import ppo, ddpg, trpo, td3
from spinup.utils.mpi_tools import mpi_fork
import tensorflow as tf
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = PATH + '/results/sawyer'
EXP_NAME = 'sawyer_reach'


def train(alg, task):
    if task == 'reach':
        env_fn = lambda: SawyerReachEnv(n_substeps=25, reward_type='dense')
    elif task == 'grasp':
        env_fn = lambda: SawyerGraspEnv(n_substeps=25, reward_type='dense')

    ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

    if alg == 'ppo':

        logger_kwargs = dict(output_dir=SAVE_PATH + '/ppo', exp_name=EXP_NAME)
        ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=3000,
            logger_kwargs=logger_kwargs, max_ep_len=500)

    elif alg == 'ddpg':

        logger_kwargs = dict(output_dir=SAVE_PATH + '/ddpg', exp_name=EXP_NAME)
        ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=1000,
             logger_kwargs=logger_kwargs, max_ep_len=200)

    elif alg == 'trpo':

        logger_kwargs = dict(output_dir=SAVE_PATH + '/trpo', exp_name=EXP_NAME)
        trpo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=2000,
             logger_kwargs=logger_kwargs, max_ep_len=200)

    elif alg == 'td3':

        logger_kwargs = dict(output_dir=SAVE_PATH + '/td3', exp_name=EXP_NAME)
        td3(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=400,
            logger_kwargs=logger_kwargs, max_ep_len=200)


def plot():
    pass


if __name__ == '__main__':
    alg = 'ppo'
    task = 'grasp'
    train(alg, task)
