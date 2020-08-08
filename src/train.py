import warnings

# src
import callbacks as clb
import configuration as C
import models
import utils

from pathlib import Path

from catalyst.dl import SupervisedRunner
from apex import amp, optimizers

def run():
    warnings.filterwarnings("ignore")
    FP16_PARAMS = dict(opt_level="O1") 

    args = utils.get_parser().parse_args()  # コマンドライン引数で参照するyamlファイルを指定
    config = utils.load_config(args.config)

    global_params = config["globals"]

    # outputディレクトリの設定
    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")  # log結果を格納

    utils.set_seed(global_params["seed"])  # seedの固定
    device = C.get_device(global_params["device"])  # CPU or GPU

    # df, datadir = C.get_metadata(config)  # original meta dataの取得
    df = C.get_resampled_metadata(config)  # resample meta dataの取得
    splitter = C.get_split(config)  # CV

    for i, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df["ebird_code"])):
        if i not in global_params["folds"]:  # 0出なければとばす(1fold-cvを意味する)
            continue

        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        
        # train/valid２種類のdata loaderをdict型で作成
        loaders = {
            phase: C.get_loader(df_, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }

        # configファイルの情報をもとにそれぞれのフレームを作成
        model = models.get_model_for_train(config).to(device)
        criterion = C.get_criterion(config).to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)
        callbacks = clb.get_callbacks(config)

        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        # catalystのクラスに渡す
        runner = SupervisedRunner(
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])

        runner.train(
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=output_dir / f"fold{i}",
            callbacks=callbacks,
            main_metric=global_params["main_metric"],
            minimize_metric=global_params["minimize_metric"],
            fp16=FP16_PARAMS)


if __name__ == "__main__":
    run()


