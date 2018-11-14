from spinup import ppo, ddpg, sac
from spinup.utils.test_policy import load_policy, run_policy
import tensorflow as tf
import sawyer_env

env_fn = lambda: sawyer_env.SawyerEnv(n_substeps=5, reward_type='dense')
save_dir = '/home/tarik/sawyer_mujoco/results/sawyer'
ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir=save_dir, exp_name='sawyer_ppo_1m')


def train():
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=800,
        logger_kwargs=logger_kwargs, target_kl=0.01, max_ep_len=500)

    # ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)


def play():
    _, get_action = load_policy(save_dir)
    env = sawyer_env.SawyerEnv()
    run_policy(env, get_action, max_ep_len=500)


if __name__ == '__main__':
    #train()
    play()