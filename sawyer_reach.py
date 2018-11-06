import os
import numpy as np
from gym.envs.robotics import rotations, robot_env

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SawyerReachEnv(robot_env.RobotEnv):

    def __init__(self, reward_type='sparse', distance_threshold=0.05, n_substeps=1):

        self.n_substeps = n_substeps
        initial_qpos = {}
        self.distance_threshold = distance_threshold
        self.nu = 7
        self.reward_type = reward_type

        super(SawyerReachEnv, self).__init__(
            model_path=MODEL_XML_PATH, n_substeps=self.n_substeps, n_actions=self.nu,
            initial_qpos=initial_qpos)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _sample_goal(self):
        return np.array([0.5, 0.2, 0.4])

    def _get_obs(self):

        right_l6_id = self.sim.model.body_name2id('right_l6')
        right_l6_pos = self.sim.data.body_xpos[right_l6_id]
        right_l6_vel = self.sim.data.body_xvelp[right_l6_id]

        arm_qpos = self.sim.data.qpos
        arm_qvel = self.sim.data.qvel

        obs = np.concatenate([right_l6_pos, right_l6_vel, arm_qpos, arm_qvel])
        return {
            'observation': obs.copy(),
            'achieved_goal': right_l6_pos.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _set_action(self, action):
        assert action.shape == (self.nu, )
        action = action.copy()

        for i in range(self.nu):
            self.sim.data.ctrl[i] = action[i]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    # def _env_setup(self, initial_qpos):
    #     pass


if __name__ == '__main__':
    sawyer_reach = SawyerReachEnv()
    # while True:
    #     sawyer_reach.render()