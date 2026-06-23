


# 🧪 AI 모델 서비스 배포 — MNIST & PillFinder

손으로 그린 숫자를 인식하는 **MNIST 분류 서비스**와, 알약 사진에서 약을 탐지하는 **PillFinder 객체 탐지 서비스**를 Streamlit으로 구현하고 Docker로 배포한 프로젝트입니다.

## 🎬 데모

https://github.com/user-attachments/assets/0427277c-33ae-4bb5-8d8c-eb0627e10dac




## 📦 구성

| | 기본 (MNIST) | 심화 (PillFinder) |
|---|---|---|
| 작업 | 손글씨 숫자 인식 | 알약 객체 탐지 (73종) |
| 모델 | ONNX MNIST-12 | YOLO11m |
| 추론 | ONNX Runtime | Ultralytics |
| UI | streamlit-drawable-canvas | 파일 업로드 / 카메라 |
| 배포 | `haneuijeong/mission17-mnist` | `haneuijeong/mission17-pill` |

## 🚀 실행

### 기본 (MNIST)
```bash
cd basic
pip install -r requirements.txt
streamlit run app.py
# 또는 Docker
docker run -p 8501:8501 haneuijeong/mission17-mnist:1.0
```

### 심화 (PillFinder)
```bash
cd advanced
pip install -r requirements.txt
streamlit run app.py
# 또는 Docker
docker run -p 8501:8501 haneuijeong/mission17-pill:1.0
```

## 🧠 모델 출처

심화의 알약 탐지 모델(YOLO11m, 73종 경구약제)은 **HealthEat 팀 프로젝트**에서 직접 학습했습니다.
데이터 전처리 파이프라인과 학습 과정은 아래 레포를 참고하세요.

➡️ [HealthEat 알약 탐지 모델 레포](https://github.com/EuijeongHan/pill_detection_project)

## 🛠 기술 스택

`Python` `Streamlit` `ONNX Runtime` `Ultralytics (YOLO11)` `PyTorch` `Docker`

## 📝 주요 구현 포인트

- **MNIST 전처리**: bounding box crop → 20px 비율 리사이즈 → 28×28 중앙 배치 → 정규화 (학습 분포 정합)
- **모델 캐싱**: `@st.cache_resource`로 세션 간 모델 재사용
- **경량 배포**: CPU 전용 PyTorch로 이미지 크기 3.1GB → 754MB 축소
- **멀티플랫폼**: `buildx`로 linux/amd64 타겟 빌드
