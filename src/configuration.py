import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms

import dataset  # src

from pathlib import Path

from criterion import Loss, Cutmix_Loss, Mixup_Loss # noqa
from transforms import (get_waveform_transforms,
                        get_spectrogram_transforms, 
                        get_spec_augment_transforms)


def get_device(device: str):
    return torch.device(device)


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get("params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


# def get_criterion(config: dict):
#     return getattr(nn, config["loss"]["name"])(**config["loss"]["params"])


# TODO 返り値の挙動
def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])


def get_metadata(config: dict):
    data_config = config["data"]
    with open(data_config["train_skip"]) as f:
        skip_rows = f.readlines()

    train = pd.read_csv(data_config["train_df_path"])
    audio_path = Path(data_config["train_audio_path"])

    for row in skip_rows:
        row = row.replace("\n", "")
        ebird_code = row.split("/")[1]
        filename = row.split("/")[2]
        train = train[~((train["ebird_code"] == ebird_code) &
                        (train["filename"] == filename))]
        train = train.reset_index(drop=True)
    return train, audio_path


# resampleされたメタデータを取得(tawata's dataset)
def get_resampled_metadata(config: dict):
    data_config = config["data"]
    tmp_list = []  # dfに変換するための一時格納リスト

    #5つのresample datasetから音声データファイルをiterationし,ebird_name,file名, file_pathをdfに変換
    # 音声データを読み込んでいるというよりかはメタデータを読み込んでいる
    for audio_d in data_config["train_resample_audio_path"]:
        audio_d = Path(audio_d)

        if not audio_d.exists():
            continue
        for ebird_d in audio_d.iterdir():
            if ebird_d.is_file():
                continue
            for wav_f in ebird_d.iterdir():
                tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])

    train_wav_path_exist = pd.DataFrame(
        tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

    del tmp_list  # 不要なのでメモリ節約のため削除

    # dfのデータにmerge
    train = pd.read_csv(Path(data_config["train_resample_audio_path"][0]) / "train_mod.csv")
    train_all = pd.merge(
        train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")

    return  train_all


# trainとvalid別々で与えられる
def get_loader(df: pd.DataFrame,
               # datadir: Path,
               config: dict,
               phase: str):
    dataset_config = config["dataset"]

    if dataset_config["name"] == "SpectrogramDataset":
        waveform_transforms = get_waveform_transforms(config)
        spectrogram_transforms = get_spectrogram_transforms(config)
        spec_augment_transforms = get_spec_augment_transforms(config)
        melspectrogram_parameters = dataset_config["params"]
        loader_config = config["loader"][phase]

        datasets = dataset.SpectrogramDataset(
            df,
            # datadir=datadir,
            img_size=dataset_config["img_size"],
            waveform_transforms=waveform_transforms,
            spectrogram_transforms=spectrogram_transforms,
            spec_augment_transforms=spec_augment_transforms,
            melspectrogram_parameters=melspectrogram_parameters)
    else:
        raise NotImplementedError

    loader = data.DataLoader(datasets, **loader_config)
    return loader

