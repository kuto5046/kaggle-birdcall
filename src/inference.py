import yaml
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import torch.utils.data as data
from fastprogress import progress_bar

# src
import callbacks as clb
import configuration as C
import models
import utils 
import dataset 


def prediction_for_clip(test_df: pd.DataFrame, 
                        clip: np.ndarray, 
                        model: ResNet, 
                        mel_params: dict, 
                        threshold=0.5):

    dataset = dataset.TestDataset(df=test_df, 
                                  clip=clip,
                                  img_size=224,
                                  melspectrogram_parameters=mel_params)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model.eval()
    prediction_dict = {}
    for image, row_id, site in progress_bar(loader):
        site = site[0]
        row_id = row_id[0]
        if site in {"site_1", "site_2"}:
            image = image.to(device)

            with torch.no_grad():
                prediction = model(image)
                proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)

            events = proba >= threshold
            labels = np.argwhere(events).reshape(-1).tolist()

        else:
            # to avoid prediction on large batch
            image = image.squeeze(0)
            batch_size = 16
            whole_size = image.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1
                
            all_events = set()
            for batch_i in range(n_iter):
                batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]
                if batch.ndim == 3:
                    batch = batch.unsqueeze(0)

                batch = batch.to(device)
                with torch.no_grad():
                    prediction = model(batch)
                    proba = prediction["multilabel_proba"].detach().cpu().numpy()
                    
                events = proba >= threshold
                for i in range(len(events)):
                    event = events[i, :]
                    labels = np.argwhere(event).reshape(-1).tolist()
                    for label in labels:
                        all_events.add(label)
                        
            labels = list(all_events)
        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: dataset.INV_BIRD_CODE[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string

    return prediction_dict


def prediction(test_df: pd.DataFrame,
               test_audio: Path,
               model_config: dict,
               mel_params: dict,
               weights_path: str,
               target_sr: int,
               logger: Optional[logging.Logger],
               threshold=0.5):

    model = get_model_for_eval(config)
    unique_audio_id = test_df.audio_id.unique()

    prediction_dfs = []
    for audio_id in unique_audio_id:
        with utils.timer(f"Loading {audio_id}", logger):
            clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),
                                   sr=TARGET_SR,
                                   mono=True,
                                   res_type="kaiser_fast")
        
        test_df_for_audio_id = test_df.query(
            f"audio_id == '{audio_id}'").reset_index(drop=True)

        with utils.timer(f"Prediction on {audio_id}", logger):
            prediction_dict = prediction_for_clip(test_df_for_audio_id,
                                                  clip=clip,
                                                  model=model,
                                                  mel_params=mel_params,
                                                  threshold=threshold)
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({
            "row_id": row_id,
            "birds": birds
        })
        prediction_dfs.append(prediction_df)
    
    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)

    return prediction_df


def run():
    TARGET_SR = 32000  # samling rate
    THRESHOLD = 0.6

    # config
    config = utils.load_config("../configs/000_ResNet50.yml")
    global_params = config["globals"]
    weights_path = global_params["weights_path"] 
    melspectrogram_parameters = config["dataset"]["params"]
    model_config = config["model"]

    # outputディレクトリの設定
    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")  # log結果を格納

    warnings.filterwarnings("ignore")
    utils.set_seed(global_params["seed"])  # seedの固定
    device = C.get_device(global_params["device"])  # CPU or GPU

    # prepare data
    data_config = config["data"]
    test = pd.read_csv(data_config["test_df_path"])
    test_audio = data_config["test_audio_path"]
    sub = pd.read_csv(data_config["sub"])
    sub.to_csv("submission.csv", index=False)  # this will be overwritten if everything goes well

    submission = prediction(test_df=test,
                            test_audio=test_audio,
                            model_config=model_config,
                            mel_params=melspectrogram_parameters,
                            weights_path=weights_path,
                            target_sr=TARGET_SR,
                            logger=logger,
                            threshold=THRESHOLD)

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    run()


