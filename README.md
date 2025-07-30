### 환경 세팅
pip install -r pyproject.toml
그리고 본인의 cuda version에 맞는 torch와 torch-geometric (+ torch-cluster, torch-sparse) 를 깔아주세요


### 모델 학습하기
```
python train.py --config {config.yaml 파일 경로}
```
이 코드를 실행하면 config.yaml 파일의 위치에 모델 베스트 체크포인트를 저장하고 자동으로 테스트셋 추론을 실시하여 Dacon 제출용 파일까지 생성합니다.

만약 체크포인트 경로를 바탕으로 추론만 하고싶다면:
```
python inference.py --config {config.yaml 파일 경로} --ckpt {ckpt 경로}
```
이 코드를 실행하면 체크포인트 바탕으로 테스트셋에 대해 추론을 실시하여 config.yaml 파일의 위치에 Dacon 제출용 파일을 생성합니다.