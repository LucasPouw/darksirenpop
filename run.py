import argparse
from config import Config
from worker import run_worker
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import get_origin, Union


def str2float(v):
    """
    Convert a string to float. If already a float or int, return as float.
    Raise ValueError if conversion fails.
    """
    if isinstance(v, (float, int)):
        return float(v)
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Cannot convert {v!r} to float")


def str2int(v):
    """
    Convert a string to int. If already an int, return it.
    Raise ValueError if conversion fails.
    Accepts '100' or '100.0'.
    """
    if isinstance(v, int):
        return v
    try:
        return int(float(v))  # auto-cast float strings to int
    except ValueError:
        raise ValueError(f"Cannot convert {v!r} to int")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y')


def str2any(v):
    if isinstance(v, (bool, float, int)):
        return v

    v_lower = v.lower()

    # Try float first
    try:
        return float(v)
    except ValueError:
        pass

    # Then bool
    if v_lower in ('true', 'yes', 'y'):
        return True
    if v_lower in ('false', 'no', 'n'):
        return False

    return v


def get_parser():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Run GW-AGN posterior analysis")

    for field_name, field_type in cfg.__annotations__.items():
        arg_name = f"--{field_name.lower()}"
        default_value = getattr(cfg, field_name)

        # Map type hints to argparse types
        if field_type == str:
            parser.add_argument(arg_name, type=field_type, default=None,
                                help=f"(default={default_value})")
        elif field_type == int:
            parser.add_argument(arg_name, type=str2int, default=None,
                                help=f"(default={default_value})")
        elif field_type == float:
            parser.add_argument(arg_name, type=str2float, default=None,
                                help=f"(default={default_value})")
        elif field_type == bool:
            parser.add_argument(arg_name, type=str2bool, default=None,
                                help=f"(default={default_value})")
        elif get_origin(field_type) is Union:
            parser.add_argument(arg_name, type=str2any, default=None,
                                help=f"(default={default_value})")
        # For arrays or objects, accept string and parse later
        else:
            parser.add_argument(arg_name, type=str, default=None,
                                help=f"(default={default_value})")

    return parser


def config_to_dict(cfg):
    result = {}
    for field in cfg.__annotations__:
        value = getattr(cfg, field)

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            value = value.tolist()

        result[field] = value

    return result


def metadata2json(fname, time_now, cfg):
    output_file = Path(cfg.OUTFILE)

    # Load existing data
    if output_file.exists():
        data = json.loads(output_file.read_text())
    else:
        data = []

    # Create entry
    entry = {
        "filename": fname,
        "timestamp": time_now,
        "config": config_to_dict(cfg)
    }

    data.append(entry)

    # Save back
    output_file.write_text(json.dumps(data, indent=2))
    return


if __name__ == "__main__":
    import time

    t = time.time()

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

    time_now = datetime.now().isoformat()
    fname = f'{cfg.POST_DIR}/{cfg.FAGN_POSTERIOR_FNAME}_{time_now}.npy'
    np.save(fname, log_llh)
    metadata2json(fname, time_now, cfg)
    print(f'Done. Posteriors are located at: {fname}\n')
    print(f'That took {(time.time() - t) / cfg.N_WORKERS} seconds/it.')