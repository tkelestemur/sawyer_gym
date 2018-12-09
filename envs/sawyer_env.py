import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = PATH + '/../model/'
GRIPPER_LINK = 'right_gripper_tip'
OBJECT = 'object'
RIGHT_GRIPPER_TIP = 'r_finger_tip'
LEFT_GRIPPER_TIP = 'l_finger_tip'


class SawyerReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, reward_type='dense', distance_threshold=0.05, n_substeps=1):

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.target_pos = np.zeros(3)
        xml_path = MODEL_PATH + 'sawyer_reach.xml'

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, n_substeps)

    def reset_model(self):

        qpos_init = np.zeros(self.sim.model.nq)
        qvel_init = np.zeros(self.sim.model.nv)
        # sample a uniform position goal
        target_offset = np.array([0.6, 0.0, 0.3])
        random_pos = np.random.uniform(low=[-0.3, -0.5, -0.4], high=[0.3, 0.5, 0.4], size=3)
        self.target_pos = target_offset + random_pos

        # move target point to random target pos
        self.sim.data.mocap_pos[0] = self.target_pos
        # qpos_init[9:12] = self.target_pos
        self.sim.nsubsteps = 25
        # initialize arm configuration and velocity
        self.set_state(qpos_init, qvel_init)
        return self._get_obs()

    def step(self, action):

        # gravity compensation
        self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]

        self.do_simulation(action, self.frame_skip)

        reward, done = self._calcuate_reach_reward(action)

        obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        eef_pos = self.sim.data.get_body_xpos(GRIPPER_LINK)
        eef_vel = self.sim.data.get_body_xvelp(GRIPPER_LINK)
        target_pos = self.target_pos
        arm_qpos = self.sim.data.qpos[:7]
        arm_qvel = self.sim.data.qvel[:7]

        obs = np.concatenate([np.sin(arm_qpos), np.cos(arm_qpos), arm_qvel, eef_pos, eef_vel, target_pos])
        return obs

    def _calcuate_reach_reward(self, action):
        # calculate reward
        if self.reward_type == 'dense':
            d = self.sim.data.get_body_xpos(GRIPPER_LINK) - self.target_pos
            euc_d = np.linalg.norm(d)

            if bool(np.abs(euc_d) < self.distance_threshold):
                reward, done = 10, True
            else:
                reward = np.exp(-0.2 * euc_d)
                done = False
            # reward = - euc_d
            # done = bool(np.abs(euc_d) < self.distance_threshold)
        elif self.reward_type == 'sparse':
            d = self.sim.data.get_body_xpos(GRIPPER_LINK) - self.target_pos
            d = np.linalg.norm(d)
            if d < self.distance_threshold:
                reward = 0.0
                done = True
            else:
                reward = -1.0
                done = False

        # r_t = - np.square(action).sum()
        # reward += r_t
        return reward, done


class SawyerGraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, reward_type='dense', n_substeps=1):

        self.reward_type = reward_type
        xml_path = MODEL_PATH + 'sawyer_grasp.xml'
        self.object_init_pos = np.array([0.7, 0, -0.1])
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, n_substeps)

    def reset_model(self):
        qpos_arm = np.array([-0.58940138, -1.1788925, 0.61659816, 1.62266692, -0.22474244, 1.2130372, -1.32163291])
        qpos_fingers = np.array([0.02083, -0.02083])
        qpos_object = np.array([0.7, 0, -0.1, 1, 0, 0, 0])
        qpos_init = np.concatenate([qpos_arm.flat, qpos_fingers.flat, qpos_object.flat])
        qvel_init = np.zeros(self.sim.model.nv)

        # initialize joint positions and velocities
        self.set_state(qpos_init, qvel_init)

        return self._get_obs()

    def step(self, action):

        # gravity compensation
        self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        # self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.do_simulation(action, self.frame_skip)

        reward_dict, done = self._calculate_grasp_reward()

        obs = self._get_obs()

        reward = reward_dict['dist_reward'] + reward_dict['grasp_reward'] + reward_dict['terminal_reward']

        return obs, reward, done, reward_dict

    def _get_obs(self):
        arm_qpos = self.sim.data.qpos[:7]
        arm_qvel = self.sim.data.qvel[:7]
        finger_qpos = self.sim.data.qpos[7:9]
        finger_qvel = self.sim.data.qvel[7:9]
        eef_pos = self.sim.data.get_body_xpos(GRIPPER_LINK)
        eef_rot = self.sim.data.get_body_xquat(GRIPPER_LINK)
        eef_vel = self.sim.data.get_body_xvelp(GRIPPER_LINK)
        object_pos = self.sim.data.get_body_xpos(OBJECT)
        object_rot = self.sim.data.get_body_xquat(OBJECT)
        object_vel = self.sim.data.get_body_xvelp(OBJECT)

        obs = np.concatenate([arm_qpos.flat, arm_qvel.flat, finger_qpos.flat, finger_qvel.flat, eef_pos.flat,
                              eef_vel.flat, eef_rot.flat, object_pos.flat, object_rot.flat, object_vel.flat])
        return obs

    def _calculate_grasp_reward(self):

        done, dist_reward, grasp_reward, terminal_reward = False, 0, 0, 0

        # calculate distance reward
        d = self.sim.data.get_site_xpos(GRIPPER_LINK) - self.sim.data.get_site_xpos(OBJECT)
        euc_d = np.linalg.norm(d)
        dist_reward = 1 - np.tanh(10.0 * euc_d)
        # dist_reward = np.exp(-0.25 * euc_d)
        # dist_reward = - euc_d

        # calculate grasp reward
        self.object_id = self.sim.model.geom_name2id(OBJECT)
        self.right_finger_id = self.sim.model.geom_name2id(RIGHT_GRIPPER_TIP)
        self.left_finger_id = self.sim.model.geom_name2id(LEFT_GRIPPER_TIP)

        right_finger_contact, left_finger_contact = False, False

        for i in range(self.sim.data.ncon):

            c = self.sim.data.contact[i]

            if c.geom1 == self.object_id and c.geom2 == self.left_finger_id:
                left_finger_contact = True
            if c.geom1 == self.left_finger_id and c.geom2 == self.object_id:
                left_finger_contact = True
            if c.geom1 == self.object_id and c.geom2 == self.right_finger_id:
                right_finger_contact = True
            if c.geom1 == self.right_finger_id and c.geom2 == self.object_id:
                right_finger_contact = True

        if left_finger_contact and right_finger_contact:

            object_pos_z = self.sim.data.get_site_xpos(OBJECT)[2] - self.object_init_pos[2]
            grasp_reward = 1 + 10 * object_pos_z

            # calculate terminal reward
            if object_pos_z > 0.1:
                terminal_reward = 100
                done = True

            # print('grasp_reward: [{}] terminal_reward: [{}]'.format(grasp_reward, terminal_reward))

        reward_dict = {'dist_reward': dist_reward, 'grasp_reward': grasp_reward,
                       'terminal_reward': terminal_reward}
        # print(reward)
        return reward_dict, done
