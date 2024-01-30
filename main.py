import torch

from feature_extractor import *

# Data

for x in dataloader:
    # Feature extraction: Dino
    feature, attention = feature_extractor(x, config)
    # PathCore
    model = AttentionCore(feature, attention, config)
    # Memory bank