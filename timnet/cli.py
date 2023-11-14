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

def train_entrypoint(args): -> None:
    cfg = OmegaConf.load(args.config_file)
    train(cfg)

def main():
    parser = get_parser()

    subparsers = parser.add_subparsers(title='subcommands',
        description='train',
        help='additional help')

    train_parser = subparsers.add_parser('train')

    train_parser.set_defaults(func=train_entrypoint)

    args = parser.parse_args()
    args.func(args)
    

if __name__ == "__main__":
    main()
