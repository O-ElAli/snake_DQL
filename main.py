import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from snake_env import SnakeEnv  # <-- import your environment class
from stable_baselines3.common.monitor import Monitor


def train():
    env = Monitor(SnakeEnv(render_mode=None))

    model = DQN("CnnPolicy", env, verbose=1, learning_rate=1e-3,
                buffer_size=50000, learning_starts=1000, batch_size=64, target_update_interval=500)

    model.learn(total_timesteps=500000)
    model.save("dqn_snake.zip")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    env.close()

def test():
    model = DQN.load("dqn_snake.zip")
    env = Monitor(SnakeEnv(render_mode="human"))
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    train()      # Run training first
    test()       # Then test the trained agent visually
