import jax

from src.environments.reaching import make, run_evals
from src.utils import save_dataframes

env, default_env_params = make()
key = jax.random.PRNGKey(0)
policy = lambda key, obs, state: env.reference_policy(key, state, 0)
eval_df = run_evals(key, policy, env, default_env_params, num_seeds=500)
save_dataframes(env.name, "reference", train_df=None, eval_df=eval_df)
