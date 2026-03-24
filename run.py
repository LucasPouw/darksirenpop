import argparse
from config import Config
from worker import run_worker
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y')


def get_parser():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Run GW-AGN posterior analysis")

    for field_name, field_type in cfg.__annotations__.items():
        arg_name = f"--{field_name.lower()}"
        default_value = getattr(cfg, field_name)

        # Map type hints to argparse types
        if field_type in [int, float, str]:
            parser.add_argument(arg_name, type=field_type, default=None,
                                help=f"(default={default_value})")
        elif field_type == bool:
            parser.add_argument(arg_name, type=str2bool, default=None,
                                help=f"(default={default_value})")
        # For arrays or objects, accept string and parse later
        else:
            parser.add_argument(arg_name, type=str, default=None,
                                help=f"(default={default_value})")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    cfg = Config()

    # Override anything provided on the CLI
    for field_name in cfg.__annotations__:
        cli_value = getattr(args, field_name.lower())
        if cli_value is not None:
            # Convert if necessary (for example, np.ndarray can be passed as comma-separated string)
            field_type = cfg.__annotations__[field_name]
            if field_type == np.ndarray:
                cli_value = np.array([float(x) for x in cli_value.split(",")])
            setattr(cfg, field_name, cli_value)

    cfg.finalize()
    
    log_llh = run_worker(cfg)

    fname = f'{cfg.POST_DIR}/{cfg.FAGN_POSTERIOR_FNAME}'
    np.save(fname, log_llh)
    print(f'Done. Posteriors are located at: {fname}.npy')
