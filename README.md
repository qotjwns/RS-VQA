# RS-VQA

인공지능프로젝트 1

## 프로젝트 구조

```text
RS-VQA/
├── baseline/
│   ├── baseline_infer.py       # count_build test split 추론, JSONL 저장
│   ├── baseline_report.py      # JSONL 결과를 CSV/PNG 리포트로 변환
│   └── configs/
│       ├── baseline.yaml       # Hydra 공통 설정
│       └── model/
│           ├── internvl/1b.yaml # InternVL3.5-1B 설정
│           ├── qwen/4b.yaml     # Qwen3.5-4B 설정
│           └── ...
├── patch_level/
│   ├── pacth_level_infer.py    # patch-level 추론 본체
│   ├── pacth_level_report.py   # patch-level 리포트 본체
│   └── configs/
│       ├── patch_level.yaml    # Hydra 공통 설정
│       └── model/              # 모델 설정(internvl/qwen)
├── model/
│   ├── download_internVL.py    # InternVL 모델 다운로드
│   └── download_Qwen.py        # Qwen 모델 다운로드
├── util/                       # baseline/patch-level 공통 유틸 패키지
├── data/                       # 로컬 데이터셋(LEVIR-MCI)
├── outputs/                    # 추론/평가 결과
├── test/
│   ├── test_infer.py           # 단일 이미지 추론 테스트 파일
│   └── test.png                # 테스트 이미지
├── requirements.txt            # 의존성 목록
└── README.md
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

## 모델 다운로드

InternVL3.5 모델 (`--size`: `1b`, `2b`, `4b`, `8b`, `14b`, 기본값 `14b`):

```bash
python3 model/download_internVL.py
python3 model/download_internVL.py --size 1b
python3 model/download_internVL.py --size 8b
```

Qwen3.5 모델 (`--size`: `0.8b`, `2b`, `4b`, `9b`, `27b`, `35b`, 기본값 `9b`):

```bash
python3 model/download_Qwen.py
python3 model/download_Qwen.py --size 0.8b
python3 model/download_Qwen.py --size 35b
```

두 스크립트 모두 `max_workers=16`으로 고정되어 있습니다.

## 빌딩 변화 개수 평가

현재 baseline은 포함된 multi-task 데이터 중 test split의 빌딩 변화 개수 task만 사용합니다.

- test: `data/coding/muti_task_data/test_task_data/count_build.json` (1,929개)

기본 설정은 `baseline/configs/baseline.yaml`에 있고, 기본 모델은 `internvl/1b`입니다.

### 추론

```bash
python3 baseline/baseline_infer.py model=internvl/14b
```

8B 모델로 실행:

```bash
python3 baseline/baseline_infer.py model=internvl/8b
python3 baseline/baseline_infer.py model=qwen/4b
```

자주 쓰는 override 예시:

```bash
python3 baseline/baseline_infer.py model=internvl/8b inference.batch_size=4 inference.limit=100
python3 baseline/baseline_infer.py model=qwen/4b inference.resume=false
```

### 리포트 생성

추론에 사용한 모델과 같은 `model` 값을 넣어 실행합니다.

```bash
python3 baseline/baseline_report.py model=internvl/14b
python3 baseline/baseline_report.py model=internvl/8b
python3 baseline/baseline_report.py model=qwen/4b
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
├── internvl3_5_14b/
│   ├── test_predictions.jsonl
│   ├── test_predictions.csv
│   ├── test_bucket_summary.csv
│   └── test_bucket_performance.png
└── ...
```

`test_predictions.jsonl`은 resume 기준 파일입니다. `inference.resume=true`이면 이미 저장된 `index`는 건너뛰고 이어서 추론합니다.

## Segmentation Evidence 시각화

`outputs/segmentation_evidence/segmentation_derived_counts.jsonl` 기반으로
overview plot과 샘플별 mask overlay figure를 생성합니다.
`label_rgb`가 있을 경우 빨간색(빌딩) 픽셀만 foreground로 사용하고,
`label`(흑백)과 교집합이 존재하면 교집합을 우선 사용합니다.
또한 시각화 시에는 JSONL의 기존 count 값을 그대로 쓰지 않고 마스크에서 component를 다시 계산합니다.

```bash
python3 patch_level/segmentation_visualize.py \
  --seg-jsonl outputs/segmentation_evidence/segmentation_derived_counts.jsonl \
  --topk-samples 8
```

patch-level 예측 결과와 같이 비교하려면:

```bash
python3 patch_level/segmentation_visualize.py \
  --seg-jsonl outputs/segmentation_evidence/segmentation_derived_counts.jsonl \
  --prediction-jsonl outputs/patch_level_building_count_test/internvl3_5_8b/test_predictions.jsonl \
  --topk-samples 8
```

출력:

- `outputs/segmentation_evidence/segmentation_evidence_overview.png` (+ `.svg`)
- `outputs/segmentation_evidence/top_error_samples/sample_XXXX.png` (+ `.svg`)

## Patch-level 빌딩 변화 개수 평가

patch-level 파이프라인도 같은 test split(`count_build.json`)을 사용합니다.

- 각 샘플에서 A/B 이미지를 `patch.grid x patch.grid`로 분할합니다.
- 각 patch에서 단일 이미지 건물 수를 추론한 뒤 patch별 `diff(B-A)`를 계산합니다.
- 최종 예측값은 유효 patch 쌍들의 `diff` 합입니다.

기본 설정은 `patch_level/configs/patch_level.yaml`에 있고, 기본 모델은 `internvl/1b`입니다.

### 추론

```bash
python3 patch_level/patch_level_infer.py model=internvl/14b
```

자주 쓰는 override 예시:

```bash
python3 patch_level/patch_level_infer.py model=internvl/8b inference.batch_size=4 inference.limit=100
python3 patch_level/patch_level_infer.py model=qwen/4b inference.resume=false
python3 patch_level/patch_level_infer.py patch.grid=2 patch.strict_valid_patches=true
```

### 리포트 생성

추론에 사용한 모델과 같은 `model` 값을 넣어 실행합니다.

```bash
python3 patch_level/patch_level_report.py model=internvl/14b
python3 patch_level/patch_level_report.py model=internvl/8b
python3 patch_level/patch_level_report.py model=qwen/4b
```

### 출력 파일

모델별 결과는 아래 경로에 따로 저장됩니다.

```text
outputs/patch_level_building_count_test/
├── internvl3_5_8b/
│   ├── test_predictions.jsonl
│   ├── test_predictions.csv
│   ├── test_bucket_summary.csv
│   └── test_bucket_performance.png
├── internvl3_5_14b/
│   ├── test_predictions.jsonl
│   ├── test_predictions.csv
│   ├── test_bucket_summary.csv
│   └── test_bucket_performance.png
└── ...
```

`test_predictions.jsonl`은 resume 기준 파일입니다. `inference.resume=true`이면 이미 저장된 `index`는 건너뛰고 이어서 추론합니다.
