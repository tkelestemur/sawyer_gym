import os
import numpy as np
from collections import namedtuple
from scipy.linalg import pinv2
import mujoco_py as mj

try:
    import spnav
except ModuleNotFoundError as exc:
    raise ImportError("Unable to load module spnav, required to interface with SpaceMouse. ") from exc

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_grasp.xml'
EEF_LINK = 'right_hand'

EEFCommand = namedtuple('EEFCommand', "grasp ctrl_mod trans_cmd rot_cmd")

class MJEEFController(object):

    def __init__(self, sim):
        # model = mj.load_model_from_path(MODEL_XML_PATH)
        # self.sim = mj.MjSim(model)
        # self.viewer = mj.MjViewer(self.sim)
        # self.sim.reset()
        # qpos_init = np.array([-0.58940138, -1.1788925, 0.61659816, 1.62266692, -0.22474244, 1.2130372, -1.32163291])
        # self.sim.data.qpos[:7] = qpos_init
        # self.sim.forward()

        self.sim = sim

        spnav.spnav_open()  # open SpaceNavigator driver
        # self.sim.model.opt.timestep = 0.003

    def get_spnav_cmd(self):
        grasp = False
        ctrl_mod = False

        sp_event = spnav.spnav_poll_event()

        if sp_event is not None and sp_event.ev_type == spnav.SPNAV_EVENT_MOTION:
            trans_cmd = np.asarray(sp_event.translation, dtype=np.float16) / 1000
            rot_cmd = np.asarray(sp_event.rotation, dtype=np.float16) / 250
            # print('pos: {}'.format(pos_cmd[0]))

        elif sp_event is not None and sp_event.ev_type == spnav.SPNAV_EVENT_BUTTON:
            # print(sp_event.bnum, sp_event.press)
            if sp_event.bnum == 0 and sp_event.press:
                grasp = not grasp
                # print('grasp: {}'.format(grasp))

            elif sp_event.bnum == 1 and sp_event.press:
                ctrl_mod = not ctrl_mod

        else:
            trans_cmd = np.zeros(3, dtype=np.float16)
            rot_cmd = np.zeros(3, dtype=np.float16)

        return EEFCommand(grasp, ctrl_mod, trans_cmd, rot_cmd)

    def controller(self, cmd):

        # desired eef velocity x, y, z, wx, wy, wz
        if cmd.ctrl_mod:
            eef_vel = np.array([0., 0., 0., cmd.rot_cmd[2], cmd.rot_cmd[0], cmd.rot_cmd[1]])
        else:
            eef_vel = np.array([cmd.trans_cmd[2], -cmd.trans_cmd[0], cmd.trans_cmd[1], 0., 0., 0.])
        # print('EEF Cmd: {} '.format(eef_vel))

        # get position jacobian of eef
        jac_pos = self.sim.data.get_body_jacp(EEF_LINK)
        jac_pos = jac_pos.reshape(3, self.sim.model.nv)
        jac_pos = jac_pos[:, 0:7]

        # get position jacobian of eef
        jac_rot = self.sim.data.get_body_jacr(EEF_LINK)
        jac_rot = jac_rot.reshape(3, self.sim.model.nv)
        jac_rot = jac_rot[:, 0:7]

        jac_full = np.concatenate((jac_pos, jac_rot))

        # calculate pseudo-inverse of jacobian
        jac_inv = pinv2(jac_full)
        q_dot = np.dot(jac_inv, eef_vel)

        # gravity compensation
        # self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]

        # self.sim.data.ctrl[:7] = q_dot
        # print('EEF Vel: {} '.format(self.sim.data.get_body_xvelp(EEF_LINK)))

        return q_dot

        # if cmd.grasp:
        #     self.sim.data.ctrl[7] = -0.020833
        #     self.sim.data.ctrl[8] = 0.020833
        # else:
        #     self.sim.data.ctrl[7] = -0.014
        #     self.sim.data.ctrl[8] = 0.014
        # self.sim.step()
        # for i in range(self.sim.data.ncon):
        #     c = self.sim.data.contact[i]
        #     # print(c.geom1, c.geom2)
        #     if c.geom1 == self.sim.model.geom_name2id('object') and c.geom2 == self.sim.model.geom_name2id('l_finger_tip'):
        #         print('contact: object and l_finger_tip')
        #     if c.geom1 == self.sim.model.geom_name2id('l_finger_tip') and c.geom2 == self.sim.model.geom_name2id('object'):
        #         print('contact: object and l_finger_tip')
        #     if c.geom1 == self.sim.model.geom_name2id('object') and c.geom2 == self.sim.model.geom_name2id('r_finger_tip'):
        #         print('contact: object and r_finger_tip')
        #     if c.geom1 == self.sim.model.geom_name2id('r_finger_tip') and c.geom2 == self.sim.model.geom_name2id('object'):
        #         print('contact: object and r_finger_tip')
        # self.viewer.render()

    # jacobian-based ik solver
    def ik(self):
        pass

if __name__ == '__main__':
    from envs.sawyer_env import SawyerReachEnv
    env = SawyerReachEnv(n_substeps=1)
    control = MJEEFController(env.sim)
    i = 0
    while True:
        if i % 5 == 0:
            cmd = control.get_spnav_cmd()
        q_dot = control.controller(cmd)
        # a = np.zeros(7)
        # a[:7] = q_dot
        env.step(q_dot)
        print('q_dot: {}'.format(q_dot))
        print('q_vel: {}'.format(env.sim.data.qvel))
        env.render()

    # control.controller()


