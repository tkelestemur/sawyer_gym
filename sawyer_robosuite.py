import os
from spinup import ppo
from spinup.utils.test_policy import load_policy
import robosuite as rs
from robosuite.wrappers import GymWrapper
PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(PATH, 'results', 'robosuite')
EXP_NAME = 'sawyer'

# Notice how the environment is wrapped by the wrapper
env = GymWrapper(
    rs.make(
        "SawyerLift",
        use_camera_obs=False,  # do not use pixel observations
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=False,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=10,  # control should happen fast enough so that simulation looks smooth
    )
)

env_fn = lambda: env
save_path = os.path.join(SAVE_PATH, 'ppo')


def train():
    logger_kwargs = dict(output_dir=save_path, exp_name=EXP_NAME)
    ppo(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=4000, epochs=10000, max_ep_len=300)


def play():
    _, get_action = load_policy(save_path)
    n_episode = 10
    ep_len, ep_ret, i = 0, 0, 0
    max_ep_len = 500
    obs = env.reset()

    while i < n_episode:
        action = get_action(obs)
        # print('Action: {}'.format(action))
        obs, r, d, reward_info = env.step(action)
        # print('Control: {}'.format(env.sim.data.ctrl))
        ep_len += 1
        ep_ret += r

        env.render()
        if d or (ep_len == max_ep_len):
            print('DONE: Episode Length: {} Episode Reward {}'.format(ep_len, ep_ret))
            obs = env.reset()
            ep_len, ep_ret, r = 0, 0, 0
            d = False
            i += 1
            # time.sleep(1)


if __name__ == "__main__":
    train()


