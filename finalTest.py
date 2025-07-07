from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv

def test_loaded_model():
    model = DQN.load("dqn_snake.zip")  # Load the model from your zip
    env = Monitor(SnakeEnv(render_mode="human"))
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    test_loaded_model()
