from snake_env import SnakeEnv
from gymnasium.utils.env_checker import check_env

env = SnakeEnv(render_mode="rgb_array")

# Validate the environment
check_env(env, skip_render_check=True)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if env.render_mode == "human":
        env.render()

env.close()
