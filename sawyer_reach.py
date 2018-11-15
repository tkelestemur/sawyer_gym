import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'


class SawyerReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, reward_type='dense', distance_threshold=0.05, n_substeps=1):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.target_pos = np.zeros(3)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, n_substeps)

    def reset_model(self):
        qpos_init = np.zeros(self.sim.model.nq)
        qvel_init = np.zeros(self.sim.model.nq)

        # sample a uniform position goal
        target_offset = np.array([0.6, 0.0, 0.3])
        random_pos = np.random.uniform(low=[-0.3, -0.5, -0.4], high=[0.3, 0.5, 0.4], size=3)
        self.target_pos = target_offset + random_pos

        # initialize arm configuration and velocity
        self.set_state(qpos_init, qvel_init)
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # calculate reward
        if self.reward_type == 'dense'
            d = self.sim.data.get_body_xpos("right_l6") - self.target_pos
            d = - np.linalg.norm(d)
        elif self.reward_type == 'sparse':
            d = self.sim.data.get_body_xpos("right_l6") - self.target_pos
            d = np.linalg.norm(d)
            if d < self.distance_threshold:
                d = 1.0
            

        # r_t = - np.square(action).sum()

        reward = d
        done = bool(np.abs(r_d) < self.distance_threshold)

        obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        eef_pos = self.sim.data.get_body_xpos('right_l6')
        eef_vel = self.sim.data.get_body_xvelp('right_l6')
        target_pos = self.target_pos
        arm_qpos = self.sim.data.qpos
        arm_qvel = self.sim.data.qvel

        obs = np.concatenate([arm_qpos, arm_qvel, eef_pos, eef_vel, target_pos])
        return obs
