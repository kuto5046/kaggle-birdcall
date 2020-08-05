import os
os.system('pip install ../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest/')
os.system('pip install efficientnet_pytorch')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import resnest.torch as resnest_torch
from efficientnet_pytorch import EfficientNet



class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


class ResNeSt(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model =getattr(resnest_torch, base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


def get_model_for_train(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if "resnet" in model_name:
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=model_params["pretrained"],
            num_classes=model_params["n_classes"])
        return model

    elif "resnest" in model_name:
        model = ResNeSt(  # type: ignore
            base_model_name=model_name,
            pretrained=model_params["pretrained"],
            num_classes=model_params["n_classes"])
        return model

    else:
        raise NotImplementedError


def get_model_for_eval(config: dict, weights_path: str):
    model_name = config["name"]
    model_params = config["params"]
    weights_path = weights_path

    if "resnet" in model_name:
        model = ResNet(
            base_model_name=model_name,
            pretrained=model_params["pretrained"],
            num_classes=model_params["n_classes"])

        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        return model
    else:
        raise NotImplementedError