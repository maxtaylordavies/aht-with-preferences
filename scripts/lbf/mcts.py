import time

import jax as jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.algos.mcts import (
    get_init_fn,
    get_recurrent_fn,
    get_interaction_loop_fn,
    run_state_names,
    get_trained_policy,
)
from src.environments.lbf import make, run_evals
from src.utils import dotdict, save_dataframes

key = jax.random.PRNGKey(0)

config = dotdict(
    {
        "num_hidden_layers": 2,
        "num_hidden_units": 128,
        "V_alpha": 0.0001,
        "pi_alpha": 0.00004,
        "b1_adam": 0.9,
        "b2_adam": 0.99,
        "eps_adam": 1e-5,
        "wd_adam": 1e-6,
        "discount": 0.99,
        "use_mixed_value": True,
        "value_scale": 0.1,
        "value_target": "maxq",
        "target_update_frequency": 100,
        "batch_size": 128,
        "avg_return_smoothing": 0.5,
        "num_simulations": 20,
        "total_steps": 5e5,
        "eval_frequency": 2e4,
        "activation": "relu",
    }
)

eval_freq_batch = config.eval_frequency // config.batch_size
opt_t, time_step = 0, 0
avg_return = jnp.zeros(config.batch_size)
episode_return = jnp.zeros(config.batch_size)
num_episodes = jnp.zeros(config.batch_size)

env, env_params = make()
num_actions = env.num_actions

key, subkey = jax.random.split(key)
(
    env_states,
    v_func,
    pi_func,
    V_opt_state,
    pi_opt_state,
    V_optim,
    pi_optim,
    V_params,
    V_target_params,
    pi_params,
) = get_init_fn(env, env_params, config)(subkey)
recurrent_fn = get_recurrent_fn(
    env, env_params, v_func, pi_func, config.batch_size, config
)
interaction_loop_fn = get_interaction_loop_fn(
    env,
    env_params,
    v_func,
    pi_func,
    V_optim,
    pi_optim,
    recurrent_fn,
    num_actions,
    eval_freq_batch,
    config,
)


var_dict = locals()
run_state = {name: var_dict[name] for name in run_state_names}

print("Beginning training...")
start_time = time.time()
times, avg_returns = [], []
pbar = tqdm(total=config.total_steps)
print(config.total_steps, config.batch_size, config.eval_frequency)
num_batches, delta_t = (
    int(config.total_steps // config.batch_size),
    int(eval_freq_batch * config.batch_size),
)
print(num_batches, delta_t)
for i in range(int(num_batches // eval_freq_batch)):
    # perform a number of iterations of agent environment interaction including learning updates
    run_state = interaction_loop_fn(run_state)

    # avg_return is debiased, and only includes batch elements with at least one completed episode so that it is more meaningful in early episodes
    valid_avg_returns = run_state["avg_return"][run_state["num_episodes"] > 0]
    valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"] > 0]
    avg_return = jnp.mean(
        valid_avg_returns / (1 - config.avg_return_smoothing**valid_num_episodes)
    )

    tqdm.write(f"Running Avg Return (t={time_step}): {avg_return}")

    avg_returns += [avg_return]
    time_step += delta_t
    times += [time_step]
    pbar.update(delta_t)

train_df = pd.DataFrame({"timestep": np.array(times), "return": np.array(avg_returns)})
elapsed = time.time() - start_time
sps = config.total_steps / elapsed
print(f"Finished training in {elapsed:.2f}s ({sps:.2f} steps/s)")

# run evals
pi_params, V_params = run_state["pi_params"], run_state["V_params"]
recurrent_fn = get_recurrent_fn(env, env_params, v_func, pi_func, 1, config)
mcts_policy = get_trained_policy(
    pi_func, v_func, pi_params, V_params, recurrent_fn, env_params, config, num_actions
)

print("Running evals...")
start_time = time.time()
eval_df = run_evals(key, mcts_policy, env, env_params, num_seeds=500)
elapsed = time.time() - start_time
print(f"Finished evals in {elapsed:.2f}s")

# save data
save_dataframes(env.name, "mcts", train_df, eval_df)

# with open("mcts.out", "wb") as f:
#     pkl.dump({"config": dict(config), "avg_returns": avg_returns, "times": times}, f)
# with open("mcts.params", "wb") as f:
#     pkl.dump({"V": V_params, "pi": pi_params}, f)
