import os
import getch
import numpy as np
from scipy.linalg import pinv2, pinv
import mujoco_py as mj
from scipy.optimize import minimize, Bounds

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_grasp.xml'
EEF_LINK = 'right_hand'

class SawyerIK(object):

    def __init__(self):
        model = mj.load_model_from_path(MODEL_XML_PATH)
        self.sim = mj.MjSim(model)
        self.viewer = mj.MjViewer(self.sim)
        self.sim.reset()
        self.sim.forward()
        self.options = {'maxiter': 10000, 'disp': True}
        self.tolerance = 1e-3
        self.bounds = Bounds(self.sim.model.jnt_range[0:7, 0],
                             self.sim.model.jnt_range[0:7, 1])

    def ik(self, eef_pos_desired=None, eef_quat_desired=None):
        q_init = self.sim.data.qpos[:7]

        if eef_pos_desired is None and eef_quat_desired is None:
            print('No position or rotation is defined!!')

        def joint_disp(q):
            return np.sum((q - q_init) ** 2)

        def pos_const(q):
            self.sim.data.qpos[:7] = q
            self.sim.forward()
            eef_pos_curr = self.sim.data.get_body_xpos('right_hand')
            # print('pos_const : {}'.format(eef_pos_curr - eef_pos_desired))
            return eef_pos_curr - eef_pos_desired

        def rot_const(q):
            self.sim.data.qpos[:7] = q
            self.sim.forward()
            eef_quat_curr = self.sim.data.get_body_xquat('right_hand')
            # print('rot_const : {}'.format(eef_quat_curr - eef_quat_desired))
            return eef_quat_curr - eef_quat_desired

        def pose_const(q):
            return np.concatenate((rot_const(q), pos_const(q)))

        # constraints = ({'type': 'eq', 'fun': pos_const},
        #                {'type': 'eq', 'fun': rot_const})

        constraints = ({'type': 'eq', 'fun': pos_const})

        q_sol = minimize(fun=joint_disp, x0=q_init, constraints=constraints, tol=self.tolerance,
                         method='SLSQP', options=self.options, bounds=self.bounds)
        print ('Solution : {}'.format(q_sol))
        self.sim.data.qpos[:7] = q_init
        self.sim.forward()

        return q_sol

    def test_ik(self):
        curr_pos = self.sim.data.get_body_xpos(EEF_LINK)
        curr_rot = self.sim.data.get_body_xquat(EEF_LINK)
        test_pos = curr_pos + np.array([-0.1, 0., 0.4])
        test_quat = curr_rot
        print('desired pos: {} \ndesired rot: {}'.format(test_pos, test_quat))

        q_sol = self.ik(test_pos, test_quat)