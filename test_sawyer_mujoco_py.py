import os
import time
import numpy as np
import mujoco_py as mj

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'

model = mj.load_model_from_path(MODEL_XML_PATH)
sim = mj.MjSim(model)

viewer = mj.MjViewer(sim)

# sim.data.qpos[0] = 0.1
# sim.forward()
# sim.model.opt.timestep = 0.001
sim.data.ctrl[:] = np.zeros(7)
sim.data.ctrl[0] = 0.1
for i in range(50000):
    sim.step()
    print ("Joint 0 Vel: {}".format(sim.data.qvel[0]))
    viewer.render()
