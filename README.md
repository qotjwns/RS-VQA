# RS-VQA

인공지능프로젝트 1

## 프로젝트 구조

```text
RS-VQA/
├── baseline/
│   ├── baseline_infer.py        # count_build test split 추론, JSONL 저장
│   ├── baseline_report.py       # JSONL 결과를 CSV/PNG 리포트로 변환
│   └── configs/
│       ├── baseline.yaml        # Hydra 공통 설정
│       └── model/
│           ├── 8b.yaml          # InternVL3.5-8B 설정
│           └── 14b.yaml         # InternVL3.5-14B 설정
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

## 빌딩 변화 개수 평가

현재 baseline은 포함된 multi-task 데이터 중 test split의 빌딩 변화 개수 task만 사용합니다.

- test: `data/coding/muti_task_data/test_task_data/count_build.json` (1,929개)

기본 설정은 `baseline/configs/baseline.yaml`에 있고, 기본 모델은 `14b`입니다.

### 추론

```bash
python3 baseline/baseline_infer.py model=14b
```

8B 모델로 실행:

```bash
python3 baseline/baseline_infer.py model=8b
```

자주 쓰는 override 예시:

```bash
python3 baseline/baseline_infer.py model=8b inference.batch_size=4 inference.limit=100
python3 baseline/baseline_infer.py model=14b inference.resume=false
```

### 리포트 생성

추론에 사용한 모델과 같은 `model` 값을 넣어 실행합니다.

```bash
python3 baseline/baseline_report.py model=14b
python3 baseline/baseline_report.py model=8b
```

### 출력 파일

모델별 결과는 서로 섞이지 않도록 아래 경로에 따로 저장됩니다.

```text
outputs/building_count_test/
├── internvl3_5_8b/
│   ├── test_predictions.jsonl
│   ├── test_predictions.csv
│   ├── test_bucket_summary.csv
│   └── test_bucket_performance.png
└── internvl3_5_14b/
    ├── test_predictions.jsonl
    ├── test_predictions.csv
    ├── test_bucket_summary.csv
    └── test_bucket_performance.png
```

`test_predictions.jsonl`은 resume 기준 파일입니다. `inference.resume=true`이면 이미 저장된 `index`는 건너뛰고 이어서 추론합니다.
