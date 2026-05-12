# RS-VQA

인공지능프로젝트 1

## 프로젝트 구조

```text
RS-VQA/
├── load_model.py          # Hugging Face에서 모델 다운로드
├── requirements.txt       # 의존성 목록
├── test/
│   ├── test_infer.py      # 단일 이미지 추론 예제
│   └── test.png           # 샘플 이미지
└── README.md
```

## 설치

```bash
python -m venv venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 모델 다운로드

아래 스크립트는 `OpenGVLab/InternVL3_5-8B-HF` 모델을 Hugging Face 캐시에 내려받습니다.

```bash
python load_model.py
```

## 추론 실행

1. `test/test_infer.py`에서 `IMAGE_PATH`(입력 이미지 경로), `PROMPT`(질문/지시문)를 환경에 맞게 수정합니다.

2. 테스트 스크립트를 실행합니다.

```bash
python test/test_infer.py
```

## 주요 파일 설명

- `load_model.py`: 모델 스냅샷 다운로드
- `test/test_infer.py`: 이미지 + 프롬프트 입력 후 텍스트 생성 테스트