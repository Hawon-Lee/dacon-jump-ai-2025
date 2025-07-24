import argparse
import yaml
import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from torch.optim import lr_scheduler
from tqdm import tqdm
from pathlib import Path
from src.dataloader import MasterDataLoader
from src.evaluate import calc_dacon_score, submit
from src.utils import load_model_instance
from inference import inference

# 예시 스크립트니까 원하는 대로 변경하여 사용하세요.


def get_instance(module, name, *args, **kwargs):
    """주어진 모듈에서 이름으로 클래스를 찾아 인스턴스화하여 반환"""
    if hasattr(module, name):
        # getattr을 사용하여 모듈에서 클래스를 가져옵니다.
        target_class = getattr(module, name)
        # 가져온 클래스를 주어진 인자와 함께 인스턴스화하여 반환합니다.
        return target_class(*args, **kwargs)
    else:
        # 클래스를 찾지 못하면 오류를 발생시킵니다.
        raise ValueError(f"모듈 {module.__name__}에서 '{name}'을(를) 찾을 수 없습니다.")


def train(
    model,
    tr_dataloader,
    vl_dataloader,
    optimizer,
    criterion,
    device,
    num_epochs=30,
    scheduler=None,
    early_stop_patience=0,
):
    """모델 학습 함수"""

    best_dacon_score = -float("inf")
    best_model_state = None
    patience_counter = 0

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

            optimizer.zero_grad()
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # 검증 단계
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            val_all_preds = []
            val_all_trues = []
            for batch_data in tqdm(vl_dataloader):
                x_in = batch_data.to(device)

                true = x_in.label.float()
                pred = model(x_in)
                pred = pred.squeeze()

                # 예측 및 정답을 기록
                val_all_preds.append(pred.cpu())
                val_all_trues.append(true.cpu())

        val_all_preds = torch.cat(val_all_preds, dim=0)
        val_all_trues = torch.cat(val_all_trues, dim=0)

        avg_val_loss = criterion(val_all_preds, val_all_trues).item()
        dacon_score = calc_dacon_score(val_all_preds, val_all_trues, metric="pIC50")
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dacon Score: {dacon_score:.4f}, LR: {current_lr:.6f}"
        )

        # 해당 에폭 성능 바탕으로 후작업
        if scheduler is not None:
            try:
                scheduler.step(dacon_score)
            except TypeError:
                # metric을 받지 않는 scheduler의 경우
                scheduler.step()

        # 최고 성능 저장 및 조기 종료 카운터 관리
        if dacon_score > best_dacon_score:
            best_dacon_score = dacon_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"INFO: New best score: {best_dacon_score:.4f}. Model saved.")
        else:
            patience_counter += 1

        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            print(f"INFO: Early stopping triggered after {epoch + 1} epochs.")
            break  # 학습 루프 탈출

    print("학습 완료!")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./experiments/hawon_exp/250722_first_exp/config.yaml",
        help="Path to the YAML configuration file",
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

    tr_dataloader = mdl.tr_dataloader()
    vl_dataloader = mdl.vl_dataloader()
    ts_dataloader = mdl.ts_dataloader()

    # model import
    model = load_model_instance(
        config["model_path"],
        config["model_name"],
        params=config.get("model_params", {}),
    )
    model.to(device)

    # optimizer initialization
    op_name = config["optimizer"]
    op_params = config.get("optimizer_params", {})
    optimizer = get_instance(optim, op_name, model.parameters(), **op_params)

    # criterion initialization
    cr_name = config["criterion"]
    criterion = get_instance(nn, cr_name)

    # scheduler initialization
    sc_name = config["scheduler"]
    sc_params = config.get("scheduler_params", {})
    scheduler = get_instance(lr_scheduler, sc_name, optimizer, **sc_params)

    # early_stop
    early_stop_patience = config["early_stop_patience"]

    # train
    best_model = train(
        model,
        tr_dataloader,
        vl_dataloader,
        optimizer,
        criterion,
        device=device,
        num_epochs=config["num_epochs"],
        scheduler=scheduler,
        early_stop_patience=early_stop_patience,
    )

    # 모델 저장 (이미 파일이 존재할 경우 뒤에 숫자 1, 2 ... 추가하여 저장)
    model_save_path = os.path.join(config_dir, "best_model.pt")
    counter = 1
    while os.path.exists(model_save_path):
        model_save_path = os.path.join(config_dir, f"best_model_{counter}.pt")
        counter += 1

    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model ckpt를 {model_save_path}에 저장하였습니다.")

    print("Inference를 시작합니다...")

    inferred = inference(best_model, ts_dataloader, device)

    submit(
        pred=inferred,
        source_csv="data/raw/sample_submission.csv",
        save_dir=config_dir,
        metric="pIC50",
    )

if __name__ == "__main__":
    main()
