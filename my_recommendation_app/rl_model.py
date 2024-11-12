import gym
from stable_baselines3 import PPO

# 簡単な環境を使用して強化学習モデルをトレーニング
env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)

def get_recommendation():
    obs = env.reset()
    action, _states = model.predict(obs)
    return f"Recommended action: {action}"