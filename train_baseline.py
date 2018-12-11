import tensorflow as tf
from baselines.ppo2 import ppo2
import robosuite as suite
from robosuite.wrappers import GymWrapper
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

env = GymWrapper(
    suite.make(
        "SawyerLift",
        use_camera_obs=False,  # do not use pixel observations
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=False,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=10,
        horizon=200
    )
)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

save_dir = "/home/tarik/sawyer_gym/results/baselines"
model_name = "sawyer_grasp_ppo"


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