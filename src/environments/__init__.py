from .reaching import make as make_reaching, run_evals as run_evals_reaching
from .lbf import make as make_lbf, run_evals as run_evals_lbf


def make(name, **kwargs):
    if name == "reaching":
        return make_reaching(**kwargs)
    elif name == "lbf":
        return make_lbf(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {name}")


def get_eval_func(name):
    if name == "reaching":
        return run_evals_reaching
    elif name == "lbf":
        return run_evals_lbf
    else:
        raise ValueError(f"Unknown environment: {name}")
