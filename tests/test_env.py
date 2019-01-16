import sys, os
import numpy as np
sys.path.append(os.path.abspath('..'))
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv
from controllers.mj_eef_controller import MJEEFController
import matplotlib.pyplot as plt

# env = SawyerReachEnv(n_substeps=1)
env = SawyerGraspEnv(n_substeps=5)
env.reset()
print('# Generalized Coordinate : {} # DoF {} # Actuators {}'.format(env.sim.model.nq, env.sim.model.nv, env.sim.model.nu))
print('QPos Init: {} \nQVel Init: {}'.format(env.sim.data.qpos, env.sim.data.qvel))


def test_eef_controller():
    control = MJEEFController(env.sim)
    while True:
        cmd = control.get_spnav_cmd()
        q_dot = control.calculate_arm_q_dot(cmd)
        gripper_q = control.calculate_gripper_q(cmd)
        a = np.zeros(9)
        a[:7] = q_dot
        a[7:9] = gripper_q

        obs, reward, done, reward_dict = env.step(a)
        # print('Observation : \n{}'.format(obs))
        print('dist_reward: {} grasp_reward: {} terminal_reward: {} '
              'total_reward: {}'.format(reward_dict['dist_reward'],
                                        reward_dict['grasp_reward'],
                                        reward_dict['terminal_reward'],
                                        reward))
        if done:
            env.reset()
        #     break
        # print('q_dot: {}'.format(q_dot))
        # print('q_vel: {}'.format(env.sim.data.qvel))
        env.render()


def test_random_controller():
    a_zero = np.zeros(env.action_space.shape[0])

    for i in range(100000):
        # a = env.action_space.sample()
        a = env.action_space.sample()
        a[:7] = np.zeros(7)
        obs, rewrad, done, _ = env.step(a_zero)
        env.render()


def test_velocity_controller():
    # control = MJEEFController(env.sim)
    # eef_vel = np.zeros(6)
    # eef_vel[2] = -0.1
    # cmd = EEFCommand(False, False, eef_vel)
    # q_dot = control.calculate_arm_q_dot(cmd)
    a = np.ones(9) * 0.5
    a[7:9] = 0.0
    T = 100
    t = np.zeros(T)
    qdot_cmd = []
    qdot_act = []
    for i in range(T):
        t[i] = i
        qdot_cmd.append(a)
        obs, rewrad, done, _ = env.step(a)
        qdot_act.append(env.sim.data.qvel[:7])
        env.render()

    plt.plot(t, qdot_act, t, qdot_cmd)
    plt.show()


if __name__ == '__main__':
    test_eef_controller()
    # test_random_controller()
    # test_velocity_controller()