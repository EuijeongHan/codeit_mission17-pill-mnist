# =============================================================================
# 미션 17 (기본) - MNIST 손글씨 숫자 인식 서비스 Docker 이미지
# -----------------------------------------------------------------------------
# 빌드 : docker build -t haneuijeong/mission17-mnist:1.0 .
# 실행 : docker run -p 8501:8501 haneuijeong/mission17-mnist:1.0
# 배포 : docker push haneuijeong/mission17-mnist:1.0
#
# 주의(Apple Silicon, M2): 멀티플랫폼 빌드 시 linux/amd64 명시 권장
#   docker buildx build --platform linux/amd64 -t haneuijeong/mission17-mnist:1.0 --push .
# =============================================================================

FROM python:3.11-slim

# 파이썬 런타임 환경 변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY app.py .

# Streamlit 기본 포트
EXPOSE 8501

# 컨테이너 헬스체크 (Streamlit 내장 health 엔드포인트)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 외부 접속 허용 설정으로 실행
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
