import time
import os
import numpy as np
import imageio
import pygame
from snake_env import SnakeEnv
from stable_baselines3 import DQN

# Load environment with rendering enabled
env = SnakeEnv(render_mode="human")

# Load trained model
model = DQN.load("dqn_snake.zip", env=env)

# Create output folder for gifs
os.makedirs("gifs", exist_ok=True)

episodes = 1
total = 0
apple_goal = 20

for i in range(episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    apples_eaten = 0
    frames = []

    while not done:
        # Capture frame from Pygame
        frame_surface = pygame.display.get_surface()
        if frame_surface:
            frame = pygame.surfarray.array3d(frame_surface)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        ep_reward += reward

        # Check if apple was eaten
        if reward >= 10:  # eating fruit gives +10
            apples_eaten += 1
            if apples_eaten >= apple_goal:
                done = True
                print(f"Reached {apple_goal} apples! Ending episode.")

    print(f"Episode {i + 1} reward: {ep_reward:.2f}")
    total += ep_reward

    # Save episode as GIF
    gif_path = f"gifs/episode_{i+1}.gif"
    if frames:
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured. GIF not saved.")

print(f"\nAverage reward over {episodes} episodes: {total / episodes:.2f}")
env.close()
