import time
from snake_env import SnakeEnv

# Create environment
env = SnakeEnv(render_mode="human")

# Reset environment and get initial observation
obs, _ = env.reset()

done = False
total_reward = 0

# Run until game ends
while not done:
    # Random action for testing (will be replaced by agent later)
    action = env.action_space.sample()

    # Take action
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Optional: Slow down so you can watch
    time.sleep(0.1)

print("Game over! Total reward:", total_reward)

# Close environment
env.close()
