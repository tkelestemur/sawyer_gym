import sys, os
import numpy as np
sys.path.append(os.path.abspath('..'))
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv
from controllers.mj_eef_controller import MJEEFController, EEFCommand
import matplotlib.pyplot as plt

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
    a_zero = np.zeros(env.action_space.shape[0])
    while True:
        # a = env.action_space.sample()
        obs, rewrad, done, _ = env.step(a_zero)
        env.render()


def test_velocity_controller():
    env.sim.nsubsteps = 1
    control = MJEEFController(env.sim)
    eef_vel = np.zeros(6)
    eef_vel[2] = -0.1
    cmd = EEFCommand(False, False, eef_vel)
    q_dot = control.calculate_arm_q_dot(cmd)
    a = np.zeros(9)
    # a[:7] = q_dot
    a[0] = 0.8
    T = 100
    t = np.zeros(T)
    qdot_cmd = np.zeros(T)
    qdot_act = np.zeros(T)
    for i in range(T):
        t[i] = i
        qdot_cmd[i] = a[0]
        obs, rewrad, done, _ = env.step(a)
        qdot_act[i] = env.sim.data.qvel[0]
        env.render()

    plt.plot(t, qdot_act, t, qdot_cmd)
    plt.show()

if __name__ == '__main__':
    # test_eef_controller()
    test_random_controller()
    # test_velocity_controller()