import os
import time
import numpy as np
import mujoco_py as mj

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/../model/sawyer_grasp.xml'

model = mj.load_model_from_path(MODEL_XML_PATH)
sim = mj.MjSim(model)

# viewer = mj.MjViewer(sim)
sim.reset()
sim.forward()

print('# Generalized Coordinate : {} # DoF {} # Actuators {}'.format(sim.model.nq, sim.model.nv, sim.model.nu))
print('# contacts {}'.format(sim.data.ncon))


# while True:
#     sim.data.qfrc_applied[:7] = sim.data.qfrc_bias[:7]
#     sim.step()
#     print('# contacts {}'.format(sim.data.ncon))
#     viewer.render()