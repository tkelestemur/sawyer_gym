import gym

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

from envs.sawyer_env import SawyerGraspEnv


# def run_task(*_):
env = SawyerGraspEnv(n_substeps=5, reward_type='dense')
# env = gym.make("InvertedDoublePendulum-v2")

env = TfEnv(normalize(env))

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))

baseline = GaussianMLPBaseline(env_spec=env.spec)

algo = PPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2048,
    max_path_length=1000,
    n_itr=488,
    discount=0.99,
    step_size=0.01,
    optimizer_args=dict(batch_size=32, max_epochs=10),
    plot=False)
algo.train()


# run_experiment(
#     run_task,
#     exp_name='sawyer_grasp',
#     n_parallel=1,
#     snapshot_mode="last",
#     seed=1,
#     plot=False,
# )