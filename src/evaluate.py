import numpy as np
import pandas as pd
import torch
from datetime import datetime

# def calc_score() -> 점수 계산하는 metric.


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
        print("입력된 값을 PIC50에서 IC50으로 변환합니다. 원치 않을 경우 convert_pIC_to_IC를 False로 변경하세요")
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
    test_array = np.random.randn(1, 127)
    submit(test_array)