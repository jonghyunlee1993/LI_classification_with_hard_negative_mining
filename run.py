import os
import yaml
import torch
import argparse

from model.model_training import ModelTraining
from training_strategy.query_false_positive import QueryFalsePositives
from training_strategy.hard_nagative_mining import HardNegativeMining

import warnings

warnings.filterwarnings(action="ignore")


def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_project_name(config):
    PROJECT_NAME = f"{config['project_name']}_fold-{config['fold_num']}"
    print("Project name:", PROJECT_NAME, "\n")

    return PROJECT_NAME


def check_device(config, args):
    if torch.cuda.is_available():
        config["device"] = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
    elif torch.backends.mps.is_available():
        config["device"] = "mps"
    else:
        config["device"] = "cpu"

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input config file path")
    parser.add_argument(
        "--gpu_no",
        default=0,
        type=int,
        help="Specify the number of gpu to utilize. Default: 0",
    )
    args = parser.parse_args()

    config_fname = args.input
    config = load_config(config_fname)
    config = check_device(config, args)

    PROJECT_NAME = get_project_name(config)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Step 1: weak model training
    weak_model_training = ModelTraining(config, PROJECT_NAME, is_weak=True)
    weak_model_training.run()

    # Step 2: query false positives
    query_false_positives = QueryFalsePositives(
        project_name=PROJECT_NAME,
        model=weak_model_training.model,
        valid_df=weak_model_training.valid_df,
        valid_transform=weak_model_training.valid_transform,
        config=config,
    ).run()

    # Step 3: hard negative mining
    hard_negative_mining = HardNegativeMining(
        project_name=PROJECT_NAME,
        model=weak_model_training.model,
        valid_transform=weak_model_training.valid_transform,
        config=config,
    ).run()

    # Step 4: strong model training
    strong_model_training = ModelTraining(config, PROJECT_NAME, is_weak=False)
    strong_model_training.run()
