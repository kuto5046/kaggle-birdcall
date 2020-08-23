import os
import sys
import warnings
from pathlib import Path
from joblib import delayed, Parallel

import librosa
import audioread
import soundfile as sf
import numpy as np
import pandas as pd
import noisereduce as nr
from tqdm import tqdm

# # define resampling function
warnings.simplefilter("ignore")


def envelope(y, rate=32000, threshold=0.25):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),min_periods=1,center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


# ノイズ除去
def denoise(y: np.ndarray):
    # ノイズを消す
    mask, _ = envelope(y)
    y_denoise = nr.reduce_noise(audio_clip=y, noise_clip=y[np.logical_not(mask)], verbose=False)
    return y_denoise


def resample(ebird_code: str, filename: str, target_sr: int):
    # 既に存在するのなら変換はしない
    print(TRAIN_DENOISED_DIR / ebird_code / filename)
    y, _ = librosa.load(TRAIN_AUDIO_DIR / ebird_code / filename, sr=target_sr, mono=True, res_type="kaiser_fast")
    y = denoise(y)
    sf.write(TRAIN_DENOISED_DIR / ebird_code / filename, y, samplerate=target_sr)


TARGET_SR = 32000
NUM_THREAD = 4  # for joblib.Parallel

TRAIN_AUDIO_DIR = Path("../input/extended-birdsong-resampled-train-audio-00")
TRAIN_DENOISED_DIR = Path("../input/extended-birdsong-resampled-denoised-train-audio-00")
train = pd.read_csv("../input/extended-birdsong-resampled-train-audio-00/train_extended.csv")

# # extract "ebird_code" and  "filename"
train_audio_infos = train[["ebird_code", "resampled_filename"]].values.tolist()

# # make directories for saving resampled audio
TRAIN_DENOISED_DIR.mkdir(parents=True, exist_ok=True)
for ebird_code in train.ebird_code.unique():
    ebird_dir = TRAIN_DENOISED_DIR / ebird_code
    ebird_dir.mkdir(exist_ok=True)


# # resample and save audio using Parallel
msg_list = Parallel(n_jobs=NUM_THREAD, verbose=1)(
    delayed(resample)(ebird_code, file_name, TARGET_SR) for ebird_code, file_name in train_audio_infos) # if not TRAIN_DENOISED_DIR / ebird_code / file_name)

# for i in tqdm(range(len(train))):
#     ebird_code, file_name = train_audio_infos[i]
#     resample(ebird_code, file_name, TARGET_SR)
#     if i == 100:
#         break

# # add information of resampled audios to train.csv
train.to_csv(TRAIN_DENOISED_DIR / "train_extended.csv", index=False)