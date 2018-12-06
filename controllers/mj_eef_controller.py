import os
import numpy as np
from collections import namedtuple
from scipy.linalg import pinv2
import mujoco_py as mj

try:
    import spnav
except ModuleNotFoundError as exc:
    raise ImportError("Unable to load module spnav, required to interface with SpaceMouse. ") from exc

EEFCommand = namedtuple('EEFCommand', "grasp ctrl_mod eef_vel")


class MJEEFController(object):

    def __init__(self, sim, eef_link='right_hand'):

        self.sim = sim
        self.eef_link = eef_link
        # open SpaceNavigator driver
        spnav.spnav_open()
        self.grasp = False  # False: open gripper True: close gripper
        self.ctrl_mod = False  # False: translation velocity True: rotational velocity

    def get_spnav_cmd(self):
        trans_cmd = np.zeros(3, dtype=np.float16)
        rot_cmd = np.zeros(3, dtype=np.float16)

        sp_event = spnav.spnav_poll_event()

        if sp_event is not None and sp_event.ev_type == spnav.SPNAV_EVENT_BUTTON:
            # print(sp_event.bnum, sp_event.press)
            if sp_event.bnum == 0 and sp_event.press:
                self.grasp = not self.grasp
                print('Gripper: {}'.format('Open' if self.grasp else 'Close'))

            elif sp_event.bnum == 1 and sp_event.press:
                self.ctrl_mod = not self.ctrl_mod
                print('Ctrl Mod: {}'.format('Translational' if not self.ctrl_mod else 'Rotational'))

        elif sp_event is not None and sp_event.ev_type == spnav.SPNAV_EVENT_MOTION:
            trans_cmd = np.asarray(sp_event.translation, dtype=np.float16) / 500
            rot_cmd = np.asarray(sp_event.rotation, dtype=np.float16) / 250
            # print('pos: {}'.format(pos_cmd[0]))

        eef_vel = np.concatenate((trans_cmd, rot_cmd))
        return EEFCommand(self.grasp, self.ctrl_mod, eef_vel)

    def calculate_gripper_q(self, cmd):
        if cmd.grasp:
            q = np.array([-0.020833, 0.020833])
        else:
            q = np.array([-0.012, 0.011])

        return q

    def calculate_arm_q_dot(self, cmd):

        # ctr_mod decides to control translational or rotational velecity
        if cmd.ctrl_mod:
            eef_vel = np.array([0., 0., 0., cmd.eef_vel[5], -cmd.eef_vel[3], cmd.eef_vel[4]])
        else:
            eef_vel = np.array([cmd.eef_vel[2], -cmd.eef_vel[0], cmd.eef_vel[1], 0., 0., 0.])
        # print('EEF Cmd: {} '.format(eef_vel))

        # get position jacobian of eef
        jac_pos = self.sim.data.get_body_jacp(self.eef_link)
        jac_pos = jac_pos.reshape(3, self.sim.model.nv)
        jac_pos = jac_pos[:, 0:7]

        # get position jacobian of eef
        jac_rot = self.sim.data.get_body_jacr(self.eef_link)
        jac_rot = jac_rot.reshape(3, self.sim.model.nv)
        jac_rot = jac_rot[:, 0:7]

        jac_full = np.concatenate((jac_pos, jac_rot))

        # calculate pseudo-inverse of jacobian
        jac_inv = pinv2(jac_full)
        q_dot = np.dot(jac_inv, eef_vel)

        return q_dot


    # jacobian-based ik solver
    def ik(self):
        pass





