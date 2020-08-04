from torchvision.transforms import transforms
import albumentations as A
import torch
import numpy as np
import random

# 波形に対する前処理
def get_waveform_transforms(config: dict):
    return None


# メルスペクトログラム（画像）に対する前処理
def get_spectrogram_transforms(config: dict):
    """
    add noise(FFT)
    mixup
    random erasing 
    holizontal flip
    """
    transforms_config = config["transforms"]
    train_transform = [
        A.HorizontalFlip(p=transforms_config["horizontal_flip"]["p"]),
        A.Cutout(p=transforms_config["cutout"]["p"]) 
    ]
    return A.Compose(train_transform)


class get_spec_augment_transforms(object):

    def __init__(self, config: dict):
        self.config = config

    def apply(self, image: np.ndarray):
        spec_augment_config = self.config["transforms"]["spec_augment"]
        image = image.copy()
        p = np.random.uniform()  # 乱数による確率
        if p <= spec_augment_config["p"]:
            for i in range(spec_augment_config["num_mask"]):
                all_frames_num, all_freqs_num = image.shape
                freq_percentage = random.uniform(0.0, spec_augment_config["freq_masking_max_percentage"])
                
                num_freqs_to_mask = int(freq_percentage * all_freqs_num)
                f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
                f0 = int(f0)
                image[:, f0:f0 + num_freqs_to_mask] = 0

                time_percentage = random.uniform(0.0, spec_augment_config["time_masking_max_percentage"])
                
                num_frames_to_mask = int(time_percentage * all_frames_num)
                t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
                t0 = int(t0)
                image[t0:t0 + num_frames_to_mask, :] = 0

        return image

# def get_spec_augment_transforms(image: np.ndarray, config: dict):
#     transforms_config = config["transforms"]
#     image = image.copy()
#     p = np.random.uniform()  # 乱数による確率
#     if p <= transforms_config["p"]:
#         for i in range(num_mask):
#             all_frames_num, all_freqs_num = image.shape
#             freq_percentage = random.uniform(0.0, transforms_config["freq_masking_max_percentage"])
            
#             num_freqs_to_mask = int(freq_percentage * all_freqs_num)
#             f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
#             f0 = int(f0)
#             image[:, f0:f0 + num_freqs_to_mask] = 0

#             time_percentage = random.uniform(0.0, transforms_config["time_masking_max_percentage"])
            
#             num_frames_to_mask = int(time_percentage * all_frames_num)
#             t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
#             t0 = int(t0)
#             image[t0:t0 + num_frames_to_mask, :] = 0

#     return image

# def mixup(input: torch.Tensor, 
#           target: torch.Tensor, 
#           gamma: float):
#     # target is onehot format!
#     perm = torch.randperm(input.size(0))
#     perm_input = input[perm]
#     perm_target = target[perm]
#     return input.mul_(gamma).add_(1 - gamma, perm_input), 
#            target.mul_(gamma).add_(1 - gamma, perm_target)

    
    