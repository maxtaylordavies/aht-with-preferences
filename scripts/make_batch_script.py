envs = ["lbf"]
algos = ["oracle", "ppo", "liam", "new_5", "new_6"]
seeds = [0, 1, 2, 3, 4]

with open("scripts/batch_script.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for env in envs:
        for algo in algos:
            for seed in seeds:
                f.write(
                    f"poetry run python scripts/train.py --env={env} --algo={algo} --seed={seed}\n"
                )
    f.write("\n")
