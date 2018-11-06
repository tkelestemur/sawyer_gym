import os
import numpy as np
import collections
from dm_control import mujoco, viewer
from dm_control.suite import base
from dm_control.rl import control

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = PATH + '/model/sawyer_mujoco.mjb'
MODEL_PATH_XML = PATH + '/model/sawyer_mujoco.xml'

_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
_DISTANCE_TOLERANCE = .1
JOINT_NAMES = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
               'right_j4', 'right_j5', 'right_j6']
LINK_NAMES = ['right_l0', 'right_l1', 'right_l2', 'right_l3',
               'right_l4', 'right_l5', 'right_l6']

class SawyerReach(base.Task):

    def __init__(self, random=None):

        super(SawyerReach, self).__init__(random=random)
        self.target_pos = np.array([0.5, 0.2, 0.4])

    def initialize_episode(self, physics):
        physics.named.data.qpos[JOINT_NAMES] = np.zeros(len(JOINT_NAMES))

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = physics.named.data.qpos[JOINT_NAMES]
        obs['qvel'] = physics.named.data.qvel[JOINT_NAMES]

        return obs

    def get_reward(self, physics):
        right_l6_pos = physics.named.data.xpos[LINK_NAMES[6]]
        distance = np.diff([self.target_pos, right_l6_pos], axis=0)
        norm = np.linalg.norm(distance)
        if norm < _DISTANCE_TOLERANCE:
            return 1.0
        else:
            return 0.0


if __name__ == '__main__':

    physics = mujoco.Physics.from_binary_path(MODEL_PATH)
    # physics = mujoco.Physics.from_xml_path(MODEL_PATH_XML)
    task = SawyerReach()
    env = control.Environment(physics, task, time_limit=_TIME_LIMIT, control_timestep=_CONTROL_TIMESTEP)

    action_spec = env.action_spec()
    time_step = env.reset()
    #
    # while not time_step.last():
    #     action = np.random.uniform(action_spec.minimum - physics.data.qfrc_bias,
    #                                action_spec.maximum - physics.data.qfrc_bias,
    #                                size=action_spec.shape)
    #     time_step = env.step(action)
    #     print(time_step.reward)

    # def random_policy(time_step):
    #     del time_step  # Unused.
    #     rand_action = np.random.uniform(low=action_spec.minimum,
    #                              high=action_spec.maximum,
    #                              size=action_spec.shape)
    #     return rand_action + physics.data.qfrc_bias
    #
    viewer.launch(env)


