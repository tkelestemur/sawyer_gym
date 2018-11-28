# Sawyer MuJoCo 

A MuJoCo and Gym based Reinforcement Learning environment using Rethink Sawyer robot for various manipulation tasks.

***This is a research code. A lot can change, break or never work.***
## Installation
*Reqiurements:*
* [MuJoCo](http://mujoco.org/)
* [mujoco-py](https://github.com/openai/mujoco-py)
* [OpenAI Gym](https://github.com/openai/gym)
* [spinup](https://github.com/openai/spinningup) (if you want to train a model)
* [hid](https://pypi.org/project/hid/) (if you want to control Sawyer's end-effector with SpaceNavigator)

Note: MuJoCo requires a license, follow the instructions [here](https://github.com/openai/mujoco-py#install-mujoco).
Anaconda 3.6 is highly recommended before you install the packages above.

## Use:
*Reaching Environment:*
```
from envs.sawyer_env import SawyerReachEnv
env = SawyerReachEnv(n_substeps=1)
env.reset()
```

*Grasping Environment:*
```
from envs.sawyer_env import SawyerGraspEnv
env = SawyerGraspEnv(n_substeps=1)
env.reset()
```

## TODO: 

- [ ] Add camera to the SawyerGraspEnv
- [ ] Fix grasp reward function
- [ ] Fix friction and penetration issues in SawyerGraspEnv

