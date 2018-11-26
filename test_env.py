import numpy as np
from envs.sawyer_env import SawyerReachEnv, SawyerGraspEnv

# env = SawyerReachEnv(n_substeps=1)
env = SawyerGraspEnv(n_substeps=1)
env.reset()

while True:
    env.step(np.zeros(env.sim.model.nu))
    env.render()