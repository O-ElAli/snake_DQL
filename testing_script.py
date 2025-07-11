import time
import os
import numpy as np
import imageio
import pygame
from datetime import datetime
from snake_env import SnakeEnv
from stable_baselines3 import DQN

# Load environment with rendering enabled
env = SnakeEnv(render_mode="human")

# Load trained model
model = DQN.load("dqn_snake.zip", env=env)

# Create output folder for gifs
os.makedirs("gifs", exist_ok=True)

episodes = 10
total_reward = 0
apple_goal = 20

for episode in range(1, episodes + 1):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    apples_eaten = 0
    frames = []

    while not done:
        # Capture frame from Pygame display surface
        frame_surface = pygame.display.get_surface()
        if frame_surface:
            frame = pygame.surfarray.array3d(frame_surface)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

            # Process pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        if reward >= 10:
            apples_eaten += 1
            print(f"Episode {episode} - Apples eaten: {apples_eaten}")
            if apples_eaten >= apple_goal:
                print(f"Reached {apple_goal} apples! Ending episode early.")
                break  # <-- Use break here instead of done=True

        time.sleep(0.05)

    total_reward += episode_reward
    print(f"Episode {episode} reward: {episode_reward:.2f}")

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_filename = f"episode_{episode}_{timestamp}.gif"
    gif_path = os.path.join("gifs", gif_filename)

    if frames:
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured, GIF not saved.")

# Print average reward
average_reward = total_reward / episodes
print(f"\nAverage reward over {episodes} episodes: {average_reward:.2f}")

# Properly close environment and pygame
env.close()
pygame.event.pump()
pygame.display.quit()
pygame.quit()
