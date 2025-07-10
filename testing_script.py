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
total_reward = 0

# Let the trained agent play
while not done:
    # Use trained model to predict action
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    time.sleep(0.1)

print("Game over! Total reward:", total_reward)
env.close()
