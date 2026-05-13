# RS-VQA

인공지능프로젝트 1

## 프로젝트 구조

```text
RS-VQA/
├── baseline/
│   ├── baseline_infer.py   # test count_build 전체 추론, JSONL 저장
│   └── baseline_report.py  # JSONL 결과를 CSV/Markdown/PNG 리포트로 변환
├── model/
│   ├── download_internVL.py # InternVL 모델 다운로드
│   └── download_Qwen.py     # Qwen 모델 다운로드
├── data/                    # 로컬 데이터셋
├── outputs/                 # 추론/평가 결과
├── test/
│   ├── test_infer.py        # 단일 이미지 추론 테스트 파일
│   └── test.png             # 테스트 이미지
├── requirements.txt         # 의존성 목록
└── README.md
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

## 모델 다운로드

InternVL 모델:

```bash
python3 model/download_internVL.py
```

Qwen 모델:

```bash
python3 model/download_Qwen.py
```

## 주요 파일 설명

- `baseline/baseline_infer.py`: `test_task_data/count_build.json`의 전체 test 샘플을 배치 추론하고 `test_predictions.jsonl`로 저장
- `baseline/baseline_report.py`: 저장된 JSONL을 읽어서 prediction CSV, bucket summary, 성능 그래프 생성
- `model/download_internVL.py`: InternVL 모델 다운로드
- `model/download_Qwen.py`: Qwen 모델 다운로드
- `test/test_infer.py`: 이미지 + 프롬프트 입력 후 텍스트 생성 테스트 파일

## 빌딩 변화 개수 평가

현재 baseline은 포함된 multi-task 데이터 중 test split의 빌딩 변화 개수 task만 사용합니다.

- test: `data/coding/muti_task_data/test_task_data/count_build.json` (1,929개)

### 1. 추론 실행

```bash
python3 baseline/baseline_infer.py
```

결과:

```text
outputs/building_count_test/test_predictions.jsonl
```

### 2. 리포트 생성

```bash
python3 baseline/baseline_report.py
```

결과:

```text
outputs/building_count_test/test_predictions.csv
outputs/building_count_test/test_bucket_summary.csv
outputs/building_count_test/test_bucket_summary.md
outputs/building_count_test/test_bucket_performance.png
```

`baseline_infer.py`의 주요 설정은 파일 상단에서 변경할 수 있습니다.

```python
MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16
LIMIT = None
RESUME = True
```
