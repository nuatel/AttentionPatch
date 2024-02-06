"""
Loads model.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits


def get_model(config, patch_size, device):

    #config.pretrain_model :
    # Initialize model with pretraining
    url = None
    if "moco" in config.pretrained_model:
        if config.pretrained_model == "moco_vit_small" and patch_size == 16:
            url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        elif config.pretrained_model == "moco_vit_base" and patch_size == 16:
            url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        model = vits.__dict__[config.pretrained_model](num_classes=0)

    elif "mae" in config.pretrained_model:
        if "mae" in config.pretrained_model and patch_size == 16:
            url = "mae/visualize/mae_visualize_vit_base.pth"
        model = vits.__dict__[config.pretrained_model](num_classes=0)
    elif "vit" in config.pretrained_model:
        if config.pretrained_model == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif config.pretrained_model == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif config.pretrained_model == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif config.pretrained_model == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif config.pretrained_model == "resnet50":
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[config.pretrained_model](patch_size=patch_size, num_classes=0)
    else:
        raise NotImplementedError
    for p in model.parameters():
        p.requires_grad = False

    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights")
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )

        if "moco" in config.pretrained_model:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in config.pretrained_model:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=True)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )
    else:
        print(
            "There is no reference weights available for this model => we use random weights."
        )

    model.eval()
    model.to(device)
    return model