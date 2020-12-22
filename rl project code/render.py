import gym
import torch
from spinup.algos.sac_pytorch.core_auto import TanhGaussianPolicySACAdapt

env = gym.make('Hopper-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0].item()

# env.reset()
# for _ in range(1000000000):
#     env.render()
#     env.step(env.action_space.sample())  # take a random action
# env.close()

policy_net = TanhGaussianPolicySACAdapt(obs_dim, act_dim, [256, 256], action_limit=act_limit)
policy_net.load_state_dict(torch.load("./model.pth"))

state = env.reset()
i, done = 0, False
for _ in range(1000000000):
    env.render()
    action = policy_net.get_env_action(state, deterministic=True)
    state, reward, done, _ = env.step(action)
    i += 1

print(f"done in {i}")
env.close()
