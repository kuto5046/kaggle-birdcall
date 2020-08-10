from torchvision.transforms import transforms
import albumentations as A
import torch
import numpy as np
import random


# 波形に対する変換
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



# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2


# def cutmix(data, targets, alpha):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_targets = targets[indices]
#  
#     lam = np.random.beta(alpha, alpha)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
#     data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
#     targets = [targets, shuffled_targets, lam]

#     return data, targets


# # バッチサイズ分のimageとlabelを入力
# def mixup(data, targets, alpha):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_targets = targets[indices]

#     lam = np.random.beta(alpha, alpha)
#     data = data * lam + shuffled_data * (1 - lam)
#     targets = [targets, shuffled_targets, lam]

#     return data, targets





    
    