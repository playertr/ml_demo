import argparse
import os
import pathlib
from omegaconf import OmegaConf
from timnet.training.train import train

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='timnet',
        description="""MNIST letter classifier."""
    )
    
    parser.add_argument('--config_file', required=False, 
        default=os.path.join(
            pathlib.Path(__file__).parent.resolve(),
            'conf/train.yaml'
        )
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    # TODO: parse subcommand verbs like "train"

    cfg = OmegaConf.load(args.config_file)
    train(cfg)

if __name__ == "__main__":
    main()
