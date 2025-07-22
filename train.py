import argparse
import yaml
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
from tqdm import tqdm
from pathlib import Path
from src.dataloader import MasterDataLoader
from src.evaluate import calc_dacon_score

# 예시 스크립트니까 원하는 대로 변경하여 사용하세요.


def load_model_instance(model_path: str, model_name: str, params: dict):
    """모델 스크립트의 경로와 모델 이름을 받아서 동적으로 모델 import하는 함수"""
    path = Path(model_path)
    # 'src/models/model.py' -> 'src.models.model'
    module_name = ".".join(path.with_suffix("").parts)

    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, model_name)
        instance = class_(**params)
        print(f"Successfully loaded class '{model_name}' from '{module_name}'")
        return instance
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_name}'. Check the path."
        ) from e
    except AttributeError:
        raise AttributeError(
            f"Could not find class '{model_name}' in module '{module_name}'."
        )


def train(
    model, tr_dataloader, vl_dataloader, optimizer, criterion, device, num_epochs=30
):
    """모델 학습 함수"""
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_data in tqdm(tr_dataloader):
            x_in = batch_data.to(device)

            true = x_in.label.float()
            pred = model(x_in)
            pred = pred.squeeze()

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 손실 계산
            loss = criterion(pred, true)

            # 역전파
            loss.backward()

            # 파라미터 업데이트
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # 검증 단계
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(vl_dataloader):
                x_in = batch_data.to(device)

                true = x_in.label.float()
                pred = model(x_in)
                pred = pred.squeeze()

                # 손실 계산
                loss = criterion(pred, true)

                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    print("학습 완료!")


def test(model, ts_dataloader, criterion, device):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./experiments/hawon_exp/250722_first_exp/config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # device 정의
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # dataloader 정의
    mdl = MasterDataLoader(
        config["train_data_dir"],
        config["data_split_file"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    
    tr_dataloader = mdl.tr_dataloader()
    vl_dataloader = mdl.vl_dataloader()

    # model import
    model = load_model_instance(
        config["model_path"],
        config["model_name"],
        params=config.get("model_params", {}),
    )
    model.to(device)
    
    # optimizer initialization
    optim_name = config["optimizer"]
    optim_params = config.get("optimizer_params", {})
    optim_params = {k:float(v) for k, v in optim_params.items()}
    if hasattr(optim, optim_name):
        optimizer = getattr(optim, optim_name)(model.parameters(), **optim_params)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_name}")

    # criterion initialization
    crit_name = config["criterion"]
    if hasattr(nn, crit_name):
        criterion = getattr(nn, crit_name)()
    else:
        raise ValueError(f"Unknown criterion name: {crit_name}")

    # train
    train(
        model,
        tr_dataloader,
        vl_dataloader,
        optimizer,
        criterion,
        device=device,
        num_epochs=config["num_epochs"],
    )

    # eval
    # test(작성 예정)

if __name__ == "__main__":
    main()