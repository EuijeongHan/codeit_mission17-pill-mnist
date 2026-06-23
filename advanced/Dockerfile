# =============================================================================
# 미션 17 (심화) - 알약 객체 탐지 서비스(PillFinder) Docker 이미지
# -----------------------------------------------------------------------------
# 빌드 : docker build -t haneuijeong/mission17-pill:1.0 .
# 실행 : docker run -p 8501:8501 haneuijeong/mission17-pill:1.0
# 배포 : docker push haneuijeong/mission17-pill:1.0
#
# 주의(Apple Silicon, M2):
#   docker buildx build --platform linux/amd64 -t haneuijeong/mission17-pill:1.0 --push .
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ultralytics(opencv) 런타임에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 + 모델 가중치 복사
COPY app.py .
COPY pill_yolo11m.pt .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
