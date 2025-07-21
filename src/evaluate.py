import numpy as np
import pandas as pd
import torch
from typing import Union
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error


def calc_dacon_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> float:

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()

    # ----------- 경진대회 평가 지표 계산 -----------
    # 컴포넌트 A: IC50(nM) 단위의 Normalized RMSE
    rmse_ic50 = root_mean_squared_error(target, pred)
    range_ic50 = np.max(target) - np.min(pred)
    norm_rmse = rmse_ic50 / range_ic50 if range_ic50 != 0 else 0
    component_A = norm_rmse

    # 컴포넌트 B: pIC50 기준 Pearson R^2
    pearson_corr, _ = pearsonr(target, pred)
    component_B = pearson_corr**2
    # 최종 점수 계산
    score = 0.4 * (1 - min(component_A, 1)) + 0.6 * component_B
    # print("\n--- 경진대회 평가 결과 ---")
    # print(f"Normalized RMSE (A): {component_A:.4f}")
    # print(f"Pearson R^2 (B): {component_B:.4f}")
    # print(f"최종 점수: {score:.4f}")

    return score


def submit(
    pred: np.ndarray | torch.Tensor,
    source_csv: str = "../data/raw/sample_submission.csv",
    save_dir: str = "../submits",
    convert_pIC_to_IC: bool = True,
) -> pd.DataFrame:

    if isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
        print("pred shape:", pred.shape)

    if len(pred.shape) == 2:
        pred = pred.ravel()

    if convert_pIC_to_IC:
        print(
            "입력된 값을 PIC50에서 IC50으로 변환합니다. 원치 않을 경우 convert_pIC_to_IC를 False로 변경하세요"
        )
        pred = 10 ** (9 - pred) / 10e9

    submission = pd.read_csv(source_csv)

    if len(pred) != submission.shape[0]:
        raise ValueError(
            f"Predicted value의 길이 {len(pred)}가 요구되는 길이 {submission.shape[0]}과 다릅니다."
        )

    now = datetime.now().strftime("%y%m%d-%H%M%S")
    path2save = f"{save_dir}/{now}.csv"
    submission["ASK1_IC50_nM"] = pred
    submission.to_csv(path2save, index=False)
    print(f"제출할 파일이 생성되었습니다. 경로: {path2save}")

    return submission


if __name__ == "__main__":
    # 테스트 코드
    pred_array = np.random.randn(1, 127)
    target_array = np.random.randn(1, 127)
    
    calc_dacon_score(pred_array, target_array)
    submit(pred_array)
