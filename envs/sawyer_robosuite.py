import os
from spinup import ppo
import robosuite as rs
from robosuite.wrappers import GymWrapper
PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(PATH, 'results')
EXP_NAME = 'sawyer'

# Notice how the environment is wrapped by the wrapper
env = GymWrapper(
    rs.make(
        "SawyerLift",
        use_camera_obs=False,  # do not use pixel observations
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=True,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=10,  # control should happen fast enough so that simulation looks smooth
    )
)


if __name__ == "__main__":
    save_path = os.path.join(SAVE_PATH, 'robosuite', 'ppo')

    logger_kwargs = dict(output_dir=save_path, exp_name=EXP_NAME)
    ppo(env_fn=env, steps_per_epoch=4000, epochs=5000,
        logger_kwargs=logger_kwargs, max_ep_len=200)


    # while True:
    #     observation = env.reset()
    #     for t in range(5000):
    #         env.render()
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         print('obs: {}'.format(observation))
    #         print()
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break