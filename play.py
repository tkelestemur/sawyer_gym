import os
import numpy as np
from spinup.utils.test_policy import load_policy
from envs.sawyer_env import SawyerReachEnv
GRIPPER_LINK = 'right_gripper_tip'

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = PATH + '/results/sawyer/ppo'

def play():
    _, get_action = load_policy(SAVE_PATH)
    env = SawyerReachEnv(n_substeps=25)
    n_episode = 100
    ep_len, ep_ret, i = 0, 0, 0
    max_ep_len = 300
    obs = env.reset()

    while i < n_episode:
        action = get_action(obs)
        # print('Action: {}'.format(action))
        obs, r, d, _ = env.step(action)
        ep_len += 1
        ep_ret += r
        env.render()
        if d or (ep_len == max_ep_len):
            d = env.sim.data.get_body_xpos(GRIPPER_LINK) - env.target_pos
            d = np.linalg.norm(d)
            print('DONE: Episode Length: {} Final Distance {}'.format(ep_len, d))
            obs = env.reset()
            ep_len, ep_ret, r = 0, 0, 0
            d = False
            i += 1
            # time.sleep(1)


if __name__ == '__main__':
    play()