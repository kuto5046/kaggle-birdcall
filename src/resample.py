import os
import warnings
from pathlib import Path
from joblib import delayed, Parallel

import librosa
import audioread
import soundfile as sf

import pandas as pd

TARGET_SR = 32000
NUM_THREAD = 4  # for joblib.Parallel

TRAIN_AUDIO_DIR = Path("../input/xeno-canto-bird-recordings-extended-a-m/A-M/")
TRAIN_RESAMPLED_DIR = Path("../input/birdsong-extended-resample-a-m/A-M/")
train = pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")

# # extract "ebird_code" and  "filename"
train_audio_infos = train[["ebird_code", "filename"]].values.tolist()

# # make directories for saving resampled audio
TRAIN_RESAMPLED_DIR.mkdir(parents=True)
for ebird_code in train.ebird_code.unique():
    ebird_dir = TRAIN_RESAMPLED_DIR / ebird_code
    ebird_dir.mkdir()

# # define resampling function
warnings.simplefilter("ignore")


def resample(ebird_code: str, filename: str, target_sr: int):

    ebird_dir = TRAIN_RESAMPLED_DIR / ebird_code

    # try:
    y, _ = librosa.load(ebird_dir / filename, sr=target_sr, mono=True, res_type="kaiser_fast")
    filename = filename.replace(".mp3", ".wav")
    sf.write(ebird_dir / filename, y, samplerate=target_sr)

    # return "OK"
    # except Exception as e:
    #     with open(resample_dir / "skipped.txt", "a") as f:
    #         file_path = str(audio_dir / ebird_code / filename)
    #         f.write(file_path + "\n")
    #     return str(e)


# # resample and save audio using Parallel
# msg_list = Parallel(n_jobs=NUM_THREAD, verbose=1)(
#     delayed(resample)(ebird_code, file_name, TARGET_SR) for ebird_code, file_name in train_audio_infos)
for ebird_code, file_name in train_audio_infos:

    resample(ebird_code, file_name, TARGET_SR)
    break

# # add information of resampled audios to train.csv
train["resampled_sampling_rate"] = TARGET_SR
train["resampled_filename"] = train["filename"].map(
    lambda x: x.replace(".mp3", ".wav"))
train["resampled_channels"] = "1 (mono)"

train.to_csv(TRAIN_RESAMPLED_DIR / "train_extended.csv", index=False)