# =============================================================================
# 미션 17 (기본) - MNIST 손글씨 숫자 인식 웹 서비스
# -----------------------------------------------------------------------------
# 사용자가 캔버스에 마우스로 숫자를 그리면, ONNX Model Zoo의 MNIST-12 모델로
# 0~9를 예측하여 확률과 함께 시각화하는 Streamlit 앱.
#
# 화면 구성 (가이드라인 4개 영역):
#   1) 입력 캔버스        : streamlit-drawable-canvas
#   2) 전처리 이미지 표시 : 모델 입력(28x28)으로 변환된 결과 시각화
#   3) 모델 추론 결과     : 0~9 각 레이블 확률 막대 차트
#   4) 이미지 저장소      : 그린 이미지 + 예측 레이블/확률 누적 표시
#
# 모델 사양 (ONNX Model Zoo MNIST-12):
#   - 입력 : "Input3",            float32, shape [1, 1, 28, 28], 0~1 정규화
#   - 출력 : "Plus214_Output_0",  float32, shape [1, 10] (softmax 미적용 logits)
#
# 전처리 핵심: 단순 28x28 축소가 아니라, MNIST 학습 분포에 맞춰
#   bounding box crop → 20px 비율 리사이즈 → 28x28 중앙 배치 → 0~1 정규화.
#   정규화를 생략하면 logits가 과도하게 커져 softmax가 한 클래스로 포화된다.
# =============================================================================

import io
import os
import hashlib
import urllib.request
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

# onnxruntime는 모델 로딩 시점에 import (불필요한 초기 import 비용 회피)
import onnxruntime as ort
from streamlit_drawable_canvas import st_canvas


# -----------------------------------------------------------------------------
# 상수 정의
# -----------------------------------------------------------------------------
# 모델 다운로드 소스 (우선순위 순). onnx/models 본 저장소는 2025-07 이후
# LFS 다운로드가 중단되어 Hugging Face 미러(onnxmodelzoo)를 1순위로 사용한다.
MODEL_URLS = [
    "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx",
    "https://huggingface.co/unity/sentis-MNIST-12/resolve/main/mnist-12.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
]
# 공식 MNIST-12 파일의 SHA256 (무결성 검증용)
MODEL_SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mnist_onnx")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist-12.onnx")

INPUT_NAME = "Input3"            # 모델 입력 텐서 이름
OUTPUT_NAME = "Plus214_Output_0" # 모델 출력 텐서 이름
CANVAS_SIZE = 280               # 입력 캔버스 한 변 픽셀 (28의 10배 → 다운샘플 안정적)


# -----------------------------------------------------------------------------
# 모델 관리: 다운로드 + 캐싱
# -----------------------------------------------------------------------------
def _sha256(path: str) -> str:
    """파일의 SHA256 해시 문자열 반환."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download_model() -> str:
    """모델을 로컬 캐시에 다운로드한다.

    - 이미 캐시에 존재하고 해시가 일치하면 재다운로드하지 않는다.
    - 여러 소스를 순서대로 시도하여 첫 성공 소스를 사용한다.
    반환: 로컬 모델 파일 경로
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) 캐시 적중: 파일이 있고 무결성 검증을 통과하면 그대로 사용
    if os.path.exists(MODEL_PATH):
        try:
            if _sha256(MODEL_PATH) == MODEL_SHA256:
                return MODEL_PATH
        except OSError:
            pass  # 손상된 캐시 → 아래에서 재다운로드

    # 2) 캐시 미스: 소스를 순서대로 시도
    last_err = None
    for url in MODEL_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            # ONNX 파일은 수십 KB 이상. LFS 포인터/에러응답(수백 바이트)이면 실패로 간주.
            if len(data) < 1000:
                raise ValueError(f"too small ({len(data)} bytes) - likely an LFS pointer or error page")
            with open(MODEL_PATH, "wb") as f:
                f.write(data)
            return MODEL_PATH
        except Exception as e:  # noqa: BLE001 - 다음 소스로 폴백
            last_err = e
            continue

    raise RuntimeError(f"모델 다운로드 실패. 마지막 오류: {last_err}")


@st.cache_resource(show_spinner="MNIST 모델을 준비하는 중...")
def load_session() -> ort.InferenceSession:
    """ONNX Runtime 추론 세션을 생성하여 반환.

    @st.cache_resource 로 데코레이트되어 있어, 한 번 생성된 세션은
    세션(브라우저 새로고침)·재실행 간에도 재사용된다(모델 캐싱).
    """
    path = _download_model()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess


# -----------------------------------------------------------------------------
# 이미지 전처리 + 추론
# -----------------------------------------------------------------------------
def preprocess(canvas_rgba: np.ndarray):
    """캔버스 RGBA 배열을 모델 입력 텐서로 변환.

    입력 : (H, W, 4) uint8. 검은 배경 + 흰색 획 (캔버스 설정과 일치).
    반환 : (tensor[1,1,28,28] float32 0~1,  vis_img 28x28 uint8 시각화용)

    MNIST 학습 데이터는 글자가 28x28 안에서 여백을 두고 중앙에 위치한다.
    이를 재현하기 위해 글자 영역을 잘라(bbox) 20px로 비율 유지 리사이즈한 뒤
    28x28 중앙에 배치하고, 픽셀값을 0~1로 정규화한다.
    """
    img = Image.fromarray(canvas_rgba.astype(np.uint8)).convert("L")  # 획=255, 배경=0
    arr = np.array(img)

    # 1) 글자 영역(흰 획) bounding box 추출
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:  # 빈 캔버스
        vis = np.zeros((28, 28), np.uint8)
        return np.zeros((1, 1, 28, 28), np.float32), vis

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = arr[y0:y1 + 1, x0:x1 + 1]

    # 2) 긴 변을 20px로 맞춰 비율 유지 리사이즈 (MNIST 관습)
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    crop_img = Image.fromarray(crop).resize((nw, nh), Image.BILINEAR)

    # 3) 28x28 캔버스 중앙에 배치
    canvas28 = Image.new("L", (28, 28), 0)
    canvas28.paste(crop_img, ((28 - nw) // 2, (28 - nh) // 2))

    vis = np.array(canvas28, dtype=np.uint8)                        # 시각화용 (0~255)
    tensor = (vis.astype(np.float32) / 255.0).reshape(1, 1, 28, 28) # 모델 입력 (0~1)
    return tensor, vis


def softmax(x: np.ndarray) -> np.ndarray:
    """수치 안정 softmax. logits(10,) → 확률(10,)."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(sess: ort.InferenceSession, tensor: np.ndarray):
    """추론 수행. 반환: (확률 배열(10,), 예측 레이블 int, 신뢰도 float)."""
    logits = sess.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0][0]  # shape (10,)
    probs = softmax(logits)
    label = int(np.argmax(probs))
    conf = float(probs[label])
    return probs, label, conf


def has_drawing(canvas_rgba: np.ndarray) -> bool:
    """캔버스에 실제로 무언가 그려졌는지 확인 (빈 캔버스 추론 방지)."""
    if canvas_rgba is None:
        return False
    # 밝기 채널에 0이 아닌 픽셀이 충분히 있으면 그림으로 간주
    gray = np.array(Image.fromarray(canvas_rgba.astype(np.uint8)).convert("L"))
    return int((gray > 10).sum()) > 5


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MNIST 손글씨 숫자 인식", page_icon="✏️", layout="wide")
    st.title("✏️ MNIST 손글씨 숫자 인식")
    st.caption("캔버스에 0~9 숫자를 그리면 ONNX MNIST-12 모델이 예측합니다.")

    # 저장소(세션 상태) 초기화
    if "gallery" not in st.session_state:
        st.session_state.gallery = []  # [{"img":PIL.Image, "label":int, "conf":float, "ts":str}, ...]

    # 모델 세션 로드 (캐시됨)
    try:
        sess = load_session()
    except Exception as e:  # noqa: BLE001
        st.error(f"모델 로딩 실패: {e}")
        st.info("네트워크에서 모델을 받지 못했습니다. 환경의 외부 접속(huggingface.co 등)을 확인하세요.")
        st.stop()

    # ----- 상단: 캔버스 / 전처리 / 추론결과 3열 -----
    col_canvas, col_pre, col_pred = st.columns([1.2, 1, 1.4])

    # (1) 입력 캔버스
    with col_canvas:
        st.subheader("1) 입력 캔버스")
        stroke = st.slider("펜 굵기", 8, 40, 18)
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke,
            stroke_color="#FFFFFF",      # 흰색 획
            background_color="#000000",  # 검은 배경 (MNIST와 동일)
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )

    drawn = canvas.image_data is not None and has_drawing(canvas.image_data)

    # (2) 전처리 이미지
    with col_pre:
        st.subheader("2) 전처리 결과")
        if drawn:
            tensor, pre_img = preprocess(canvas.image_data)  # 텐서 + 시각화 이미지
            st.image(pre_img, caption="28×28 grayscale (모델 입력)", width=160)
        else:
            st.info("왼쪽 캔버스에 숫자를 그려 주세요.")

    # (3) 추론 결과
    with col_pred:
        st.subheader("3) 추론 결과")
        if drawn:
            probs, label, conf = predict(sess, tensor)
            st.metric("예측 숫자", f"{label}", f"{conf*100:.1f}%")
            # 0~9 확률 막대 차트
            st.bar_chart({"확률": {str(i): float(probs[i]) for i in range(10)}})
        else:
            st.info("그림이 입력되면 0~9 확률이 표시됩니다.")

    # ----- 저장 버튼 -----
    st.divider()
    if drawn:
        if st.button("💾 현재 이미지와 예측을 저장소에 추가"):
            # 캔버스 원본을 보기 좋은 크기로 저장
            disp = Image.fromarray(canvas.image_data.astype(np.uint8)).convert("L")
            st.session_state.gallery.insert(0, {
                "img": disp,
                "label": label,
                "conf": conf,
                "ts": datetime.now().strftime("%H:%M:%S"),
            })
            st.success(f"저장됨: 예측 {label} ({conf*100:.1f}%)")

    # (4) 이미지 저장소
    st.subheader("4) 이미지 저장소")
    if not st.session_state.gallery:
        st.caption("저장된 이미지가 없습니다. 위에서 그린 뒤 '저장소에 추가'를 눌러 보세요.")
    else:
        if st.button("🗑️ 저장소 비우기"):
            st.session_state.gallery = []
            st.rerun()
        cols = st.columns(6)
        for idx, item in enumerate(st.session_state.gallery):
            with cols[idx % 6]:
                st.image(item["img"], width=80)
                st.caption(f"#{item['label']} · {item['conf']*100:.0f}%\n{item['ts']}")


if __name__ == "__main__":
    main()
