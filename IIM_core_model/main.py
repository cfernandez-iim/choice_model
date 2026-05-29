import argparse
import logging
import yaml

from IIM_core_model.utils.example import help
from IIM_core_model.utils.logger import get_logger
from IIM_core_model.utils.params_loading import initialize_seg_params

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base, new):
    """Recursively merge two dictionaries"""
    for key, value in new.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def main():
    parser = argparse.ArgumentParser(description="My App")

    # Allow multiple --config arguments
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to config file (can be used multiple times)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = get_logger()
    logger.info("Starting IIM_model")

    final_config = {}

    for config_path in args.config:
        cfg = load_yaml(config_path)
        final_config = merge_dicts(final_config, cfg)

    seg_params = initialize_seg_params(final_config['segment_csv'])

    print(seg_params)
    help()

if __name__ == "__main__":
    main()