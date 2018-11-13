import os
import mujoco_py as mj

PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = PATH + '/model/sawyer_reach.xml'

model = mj.load_model_from_path(MODEL_XML_PATH)
sim = mj.MjSim(model)

viewer = mj.MjViewer(sim)

while True:
    sim.step()
    viewer.render()
