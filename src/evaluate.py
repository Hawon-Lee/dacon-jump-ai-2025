import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from typing import Union, Literal, Optional


def calc_dacon_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    metric: Literal["pIC50", "IC50_nM", "IC50_M"] = "pIC50",
) -> float:

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()

    # IC50(nM) 과 pIC50 으로 통일
    if metric == "pIC50":
        pred_nM, target_nM = 10 ** (9 - pred), 10 ** (9 - target)
        pred_p, target_p = pred, target
    elif metric == "IC50_nM":
        pred_nM, target_nM = pred, target
        pred_p, target_p = -np.log10(pred / 1e9), -np.log10(target / 1e9)
    elif metric == "IC50_M":
        pred_nM, target_nM = pred * 1e9, target * 1e9
        pred_p, target_p = -np.log10(pred), -np.log10(target)

    # ----------- 경진대회 평가 지표 계산 -----------
    # 컴포넌트 A: IC50(nM) 단위의 Normalized RMSE
    rmse_ic50 = root_mean_squared_error(target_nM, pred_nM)
    range_ic50 = np.max(target_nM) - np.min(target_nM)
    norm_rmse = rmse_ic50 / range_ic50 if range_ic50 != 0 else 0
    component_A = norm_rmse

    # 컴포넌트 B: pIC50 기준 Pearson R^2
    pearson_corr, _ = pearsonr(target_p, pred_p)
    component_B = pearson_corr**2
    # 최종 점수 계산
    score = 0.4 * (1 - min(component_A, 1)) + 0.6 * component_B
    # print("\n--- 경진대회 평가 결과 ---")
    # print(f"Normalized RMSE (A): {component_A:.4f}")
    # print(f"Pearson R^2 (B): {component_B:.4f}")
    # print(f"최종 점수: {score:.4f}")

    return score


def submit(
    pred: Union[np.ndarray, torch.Tensor],
    source_csv: str = "../data/raw/sample_submission.csv",
    save_dir: str = "../submits",
    metric: Literal["pIC50", "IC50_nM", "IC50_M"] = "pIC50",
    suffix: Optional[str] = ""
) -> pd.DataFrame:

    print("예측된 IC50값 shape:", pred.shape)

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()

    if len(pred.shape) == 2:
        pred = pred.ravel()

    # IC50(nM) 으로 통일
    if metric == "pIC50":
        pred_nM = 10 ** (9 - pred)
    elif metric == "IC50_nM":
        pred_nM = pred
    elif metric == "IC50_M":
        pred_nM = pred * 1e9

    submission = pd.read_csv(source_csv)

    if len(pred_nM) != submission.shape[0]:
        raise ValueError(
            f"predicted value의 길이 {len(pred_nM)}가 요구되는 길이 {submission.shape[0]}과 다릅니다."
        )

    now = datetime.now().strftime("%y%m%d-%H%M%S")
    if suffix:
        now += f"_{suffix}"
    path2save = f"{save_dir}/{now}.csv"
    submission["ASK1_IC50_nM"] = pred_nM
    submission.to_csv(path2save, index=False)
    print(f"제출할 파일이 생성되었습니다. 경로: {path2save}")

    return submission


if __name__ == "__main__":
    # 테스트 코드
    pred_array = np.random.randn(1, 127)
    target_array = np.random.randn(1, 127)

    calc_dacon_score(pred_array, target_array)
    submit(pred_array)
