import sys, os
import numpy as np
sys.path.append(os.path.abspath('..'))
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv
from controllers.mj_eef_controller import MJEEFController


# env = SawyerReachEnv(n_substeps=1)
env = SawyerGraspEnv(n_substeps=1)
env.reset()
print('# Generalized Coordinate : {} # DoF {} # Actuators {}'.format(env.sim.model.nq, env.sim.model.nv, env.sim.model.nu))


def test_eef_controller():
    control = MJEEFController(env.sim)
    i = 0
    while True:
        cmd = control.get_spnav_cmd()
        q_dot = control.calculate_arm_q_dot(cmd)
        gripper_q = control.calculate_gripper_q(cmd)
        a = np.zeros(9)
        a[:7] = q_dot
        a[7:9] = gripper_q

        obs, reward, done, _ = env.step(a)
        if done:
            break
        # print('q_dot: {}'.format(q_dot))
        # print('q_vel: {}'.format(env.sim.data.qvel))
        env.render()


def test_random_controller():

    while True:
        a = env.action_space.sample()
        obs, rewrad, done, _ = env.step(a)
        env.render()


if __name__ == '__main__':
    # test_eef_controller()
    test_random_controller()