from hwtmode.data import load_patch_files
import argparse
import yaml
from os.path import exists, join


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, yaml.Loader)
    # Load training data
    train_input, train_output, train_meta = load_patch_files()
    # Load validation data

    # Load testing data

    # Transform
    return


if __name__ == "__main__":
    main()
