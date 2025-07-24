import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataloader import MasterDataLoader
from src.evaluate import submit
from src.utils import load_model_instance


def inference(model, ts_dataloader, device) -> torch.Tensor:
    """모델 평가 함수"""

    model.eval()
    with torch.no_grad():
        all_preds = []
        for batch_data in tqdm(ts_dataloader):
            x_in = batch_data.to(device)

            pred = model(x_in)
            pred = pred.squeeze()

            # 예측 및 정답을 기록
            all_preds.append(pred.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./experiments/hawon_exp/250722_first_exp/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./experiments/hawon_exp/250722_first_exp/ckpt.pt",
        help="Path to the model checkpoint file",
    )
    args = parser.parse_args()

    config_dir = os.path.dirname(args.config)
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # device 정의
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # dataloader 정의
    mdl = MasterDataLoader(
        config["train_data_dir"],
        config["test_data_dir"],
        config["data_split_file"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    ts_dataloader = mdl.ts_dataloader()

    # model import
    model = load_model_instance(
        config["model_path"],
        config["model_name"],
        params=config.get("model_params", {}),
    )
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.to(device)
    print(f"Successfully load best ckpt")

    # train
    pred = inference(model, ts_dataloader, device)

    # submit file generation
    submit(
        pred=pred,
        source_csv="data/raw/sample_submission.csv",
        save_dir=config_dir,
        metric="pIC50",
    )

if __name__ == "__main__":
    main()