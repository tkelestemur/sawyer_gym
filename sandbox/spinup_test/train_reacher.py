from spinup import ppo
import tensorflow as tf
import gym


env_fn = lambda: gym.make('Reacher-v2')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='/home/tarik/sawyer_mujoco/results/spinup', exp_name='reacher_ppo_1m')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)