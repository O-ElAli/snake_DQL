import time
from snake_env import SnakeEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
# Load environment with rendering enabled
env = SnakeEnv(render_mode="human")

# Load trained model
model = DQN.load("dqn_snake.zip", env=env)  # Make sure this matches your saved model name

# Reset environment
obs, _ = env.reset()
done = False
total = 0
episodes = 10
for i in range(episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        ep_reward += reward
    print(f"Episode {i + 1} reward: {ep_reward:.2f}")
    total += ep_reward

print(f"\nAverage reward over {episodes} episodes: {total / episodes:.2f}")

env.close()
