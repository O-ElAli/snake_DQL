import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv

MODEL_PATH = "dqn_snake"
timestamps = 2000000  # Default total timesteps for training

def train(total_timesteps=timestamps):
    env = Monitor(SnakeEnv(render_mode=None))

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Found saved model — resuming training...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("No saved model found — starting fresh training...")
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3,
                    buffer_size=50000, learning_starts=1000,
                    batch_size=64, target_update_interval=500)

    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

def test():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print("No trained model found. Train the model first.")
        return

    model = DQN.load(MODEL_PATH)
    env = Monitor(SnakeEnv(render_mode="human"))  # Rendering is handled internally here

    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    env.close()

if __name__ == "__main__":
    train(total_timesteps=timestamps)
    test()
