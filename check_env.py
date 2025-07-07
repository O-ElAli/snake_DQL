from snake_env import SnakeEnv
from gymnasium.utils.env_checker import check_env

# Instantiate environment
env = SnakeEnv(render_mode="rgb_array")  # Use rgb_array for headless check

# Validate it
check_env(env, skip_render_check=True)
