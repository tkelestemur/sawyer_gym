import os
import numpy as np
from spinup.utils.test_policy import load_policy, run_policy
from sawyer_reach import SawyerReachEnv

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = PATH + '/results/sawyer'

def play():
    _, get_action = load_policy(SAVE_PATH)
    env = SawyerReachEnv()
    n_episode = 100
    ep_len, ep_ret, i = 0, 0, 0
    max_ep_len = 300
    obs = env.reset()

    while i < n_episode:
        env.render()
        action = get_action(obs)
        obs, r, d, _ = env.step(action)
        ep_len += 1
        ep_ret += r
        if d or (ep_len == max_ep_len):
            d = env.sim.data.get_body_xpos("right_l6") - env.target_pos
            d = np.linalg.norm(d)
            print('DONE: Episode Length: {} Final Distance {}'.format(ep_len, d))
            obs = env.reset()
            ep_len, ep_ret, r = 0, 0, 0
            d = False
            i += 1


if __name__ == '__main__':
    play()