import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'


class SawyerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, reward_type='dense', distance_threshold=0.05, n_substeps=2):

        # self.reward_type = reward_type
        # self.distance_threshold = distance_threshold

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, n_substeps)

    def reset_model(self):
        return self._get_obs()

    def step(self, action):

        # calculate reward

        # right_l6_id = self.sim.model.body_name2id('right_l6')
        # right_l6_pos = self.sim.data.body_xpos[right_l6_id]
        d = self.get_body_com('right_l6') - self.get_body_com('target')
        r_d = - np.linalg.norm(d)
        r_t = - np.square(action).sum()

        reward = r_d + r_t
        # if self.reward_type == 'sparse':
        # reward = -(r_d > self.distance_threshold).astype(np.float32)
        # else:
        #     reward = -d

        self.do_simulation(action, self.frame_skip)

        # done = (d < self.distance_threshold).astype(np.float32)
        # done = (r_d < 0.05).astype(np.float32)
        done = False
        # done = False
        # get observation
        obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        eef_pos = self.sim.data.get_body_xpos('right_l6')
        eef_vel = self.sim.data.get_body_xvelp('right_l6')
        d = self.get_body_com('right_l6') - self.get_body_com('target')
        arm_qpos = self.sim.data.qpos
        arm_qvel = self.sim.data.qvel

        obs = np.concatenate([arm_qpos, arm_qvel, eef_pos, eef_vel, d])
        return obs

