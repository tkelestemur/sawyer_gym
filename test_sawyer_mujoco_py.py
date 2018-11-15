import os
import time
import numpy as np
import mujoco_py as mj

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'

model = mj.load_model_from_path(MODEL_XML_PATH)
sim = mj.MjSim(model)

viewer = mj.MjViewer(sim)
sim.reset()

sim.data.body_xpos[9] = np.array([1, 0, 1])
sim.forward()
# qpos = np.zeros(7)
# qvel = np.zeros(7)
# qpos[1] = 0.5
# old_state = sim.get_state()
# new_state = mj.MjSimState(old_state.time, qpos, qvel,
#                                  old_state.act, old_state.udd_state)
# sim.set_state(new_state)
# sim.forward()
#


while True:
    sim.step()
    viewer.render()