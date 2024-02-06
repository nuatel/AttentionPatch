import argparse
from omegaconf import OmegaConf
import os
import numpy as np
import torch
from networks import get_model


def tuning_model(config):
    # output: the mode_weight

def anomaly_detection(config):
    # save anomaly map, pre, score to result

def train(config):
    torch.manual_seed(42)
    np.random.seed(42)
    model = get_model(config)
    print("Num params: ", sum(p.numel() for p in model.paramters()))
    model = model.to(config.model.device)

def parse_args():
    parser = argparse.ArgumentParser("AttentionPatch")
    parser.add_argument('cfg', '--config',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
                        help='config file')
    parser.add_argument('--tuning',
                        default=False,
                        help='Tuning model')
    parser.add_argument('--train',
                        default=True,
                        help='Train and build memory bank')
    parser.add_argument('--detection',
                        default=False,
                        help='Detect anomaly')
    args, unknowns = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    print("Datasets: ", config.data.name)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.tuning:
        print("Tuning ...")
        tuning_model(config)

    if args.training:
        print("Training...")
        train(config)

    if args.detection:
        print("Detecting ...")
        anomaly_detection(config)


