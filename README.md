# RS-VQA

인공지능프로젝트 1

## 프로젝트 구조

```text
RS-VQA/
├── load_model.py          # Hugging Face에서 모델 다운로드
├── requirements.txt       # 의존성 목록
├── test/
│   ├── test_infer.py      # vast.ai 구동용 테스트 파일
│   └── test.png           # 테스트 이미지
└── README.md
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

## 모델 다운로드

아래 스크립트는 `InternVL3_5-8B` 모델을 내려받습니다.

```bash
python load_model.py
```

## 주요 파일 설명

- `load_model.py`: 모델 스냅샷 다운로드
- `test/test_infer.py`: 이미지 + 프롬프트 입력 후 텍스트 생성 테스트 파일