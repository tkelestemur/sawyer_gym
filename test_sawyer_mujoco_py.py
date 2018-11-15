import os
import time
import numpy as np
import mujoco_py as mj

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'

model = mj.load_model_from_path(MODEL_XML_PATH)
sim = mj.MjSim(model)

# viewer = mj.MjViewer(sim)
sim.reset()
# qpos = np.zeros(7)
# qvel = np.zeros(7)
# qpos[1] = 0.5
# old_state = sim.get_state()
# new_state = mj.MjSimState(old_state.time, qpos, qvel,
#                                  old_state.act, old_state.udd_state)
# sim.set_state(new_state)
# sim.forward()
#
# sim.data.ctrl[:] = np.zeros(7)
# sim.data.ctrl[1] = 0.1
# i = 0
# for i in range(20):
#
#     sim.step()
#     i += 1
#     viewer.render()
#     print ("Joint 1 Vel: {}".format(sim.data.qvel[1]))

# while True:
#     sim.step()
#     viewer.render()