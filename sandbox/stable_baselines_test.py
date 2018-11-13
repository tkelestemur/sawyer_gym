import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make("Reacher-v2")])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

# Don't forget to save the running average when saving the agent
# log_dir = "/tmp/"
# model.save(log_dir + "ppo_reacher")
# env.save_running_average(log_dir)


obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones[0]:
        print("Done reward: {}".format(rewards))
        time.sleep(1)
    env.render()