import sys, os
import numpy as np
sys.path.append(os.path.abspath('..'))
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv
from controllers.mj_eef_controller import MJEEFController, EEFCommand
import matplotlib.pyplot as plt

# env = SawyerReachEnv(n_substeps=1)
env = SawyerGraspEnv(n_substeps=1)
env.reset()


def test_gripper():
    l_gripper_pos = np.linspace(0.0,
                                0.02, 100)
    r_gripper_pos = np.linspace(0.0,
                                0.02, 100)



    a = np.zeros(9)
    j = 0
    for i in range(10000000):
        if i % 100 == 0:

            # a[7] = l_gripper_pos[j]
            # a[8] = r_gripper_pos[j]
            # a[7] = 0.02
            # a[8] = -0.02

            j += 1
        # env.step(a)
        env.render()
        if i % 100 == 0:
            print('finger a    : {}'.format(a[7:9]))
            print('finger qpos : {}'.format(env.sim.data.qpos[7:9]))


if __name__ == '__main__':
    test_gripper()