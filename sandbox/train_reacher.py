import os
import gym
import time
import numpy as np
# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None

# from baselines import logger
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.cmd_util import make_mujoco_env
from baselines.ddpg import ddpg
from baselines.ppo2 import ppo2
from baselines.common import models

ENV_ID = "Reacher-v2"

env = make_mujoco_env(ENV_ID, 123)
# gym_env = gym.make(ENV_ID)
# mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
# env = Monitor(env,
#               logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
#               allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)


save_dir = "/home/tarik/sawyer_mujoco/data/"
model_name = "reach_ppo2_1m"
network = 'mlp'

def train(save=False):

    model = ppo2.learn(network=network, env=env, total_timesteps=100000, nminibatches=32, lam=0.95,
                       gamma=0.99, noptepochs=10, log_interval=1, ent_coef=0)

    if save:
        model.save(save_dir + model_name)
        # env.save_running_average(save_dir)

def play():
    # model = ppo2.learn(network='mlp', env=env, total_timesteps=0,
    # load_path=save_dir + model_name)
    #
    # obs = env.reset()
    #
    # def initialize_placeholders(nlstm=128, **kwargs):
    #     return np.zeros((1, 2 * nlstm)), np.zeros((1))
    #
    # state, dones = initialize_placeholders()
    # while True:
    #     actions, _, state, _ = model.step(obs, S=state, M=dones)
    #     obs, _, done, _ = env.step(actions)
    #     done = done.any() if isinstance(done, np.ndarray) else done
    #
    #     if done:
    #         time.sleep(3)
    #         obs = env.reset()
    #     env.render()
    print(env.reset())


if __name__ == '__main__':
    train(True)
    # play()