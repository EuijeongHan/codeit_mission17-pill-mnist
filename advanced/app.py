# =============================================================================
# 미션 17 (심화) - 알약 객체 탐지 웹 서비스 (PillFinder)
# -----------------------------------------------------------------------------
# 사용자가 휴대폰으로 촬영하거나 업로드한 알약 사진에서, YOLO11m 모델로
# 73종 알약을 탐지하여 위치(박스) · 품목명 · 신뢰도를 시각화한다.
#
# - 기반 모델 : HealthEat 팀 프로젝트에서 학습한 YOLO11m (73 classes, imgsz=800)
# - 추론 엔진 : ultralytics (letterbox 전처리 + NMS 후처리 내장)
#   → 휴대폰 사진의 임의 해상도/비율/회전을 자동 대응하므로 별도 전처리 불필요
#
# 화면 구성:
#   1) 이미지 입력  : 파일 업로드 또는 카메라 촬영
#   2) 탐지 결과 이미지 : 바운딩 박스 + 레이블이 그려진 이미지
#   3) 탐지 목록    : 알약별 품목명 · 신뢰도 표
# =============================================================================

import io
import os
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------
# 가중치 경로: 환경변수로 덮어쓸 수 있게 하되 기본은 동봉된 파일
MODEL_PATH = os.environ.get("PILL_MODEL_PATH", "pill_yolo11m.pt")
IMG_SIZE = 800          # 학습 시 imgsz와 동일하게 추론
DEFAULT_CONF = 0.25     # 신뢰도 임계값 기본값
DEFAULT_IOU = 0.45      # NMS IoU 임계값


# -----------------------------------------------------------------------------
# 모델 관리: 로드 + 캐싱
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="알약 탐지 모델을 불러오는 중...")
def load_model(path: str) -> YOLO:
    """YOLO 모델을 로드하여 반환.

    @st.cache_resource 로 캐싱되어, 모델 가중치(약 39MB)는 최초 1회만
    메모리에 로드되고 이후 재실행·세션 간 재사용된다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    return YOLO(path)


# -----------------------------------------------------------------------------
# 추론
# -----------------------------------------------------------------------------
def run_detection(model: YOLO, pil_img: Image.Image, conf: float, iou: float):
    """이미지에서 알약을 탐지.

    반환:
      - result_img : 박스가 그려진 결과 이미지 (PIL.Image)
      - detections : [{"name":품목명, "conf":신뢰도, "xyxy":[x1,y1,x2,y2]}, ...]
    ultralytics가 내부적으로 letterbox 리사이즈 + 추론 + NMS를 수행한다.
    """
    rgb = pil_img.convert("RGB")
    # predict: imgsz로 letterbox 후 추론, conf/iou로 필터 + NMS
    result = model.predict(
        source=np.array(rgb),
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    # 박스가 그려진 결과 이미지. plot(rgb=True)로 RGB를 직접 받아 색 반전을 방지
    plotted = result.plot()                 # 기본 BGR(ndarray)
    result_img = Image.fromarray(plotted)  # BGR → RGB

    # 탐지 목록 구성
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "name": model.names[cls_id],
            "conf": float(box.conf[0]),
            "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    # 신뢰도 높은 순 정렬
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return result_img, detections


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="알약 탐지 - PillFinder", page_icon="💊", layout="wide")
    st.title("💊 PillFinder — 알약 객체 탐지")
    st.caption("휴대폰으로 찍거나 업로드한 사진에서 73종 알약을 탐지합니다. (YOLO11m 기반)")

    # 모델 로드
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        st.error(f"모델 로딩 실패: {e}")
        st.stop()

    # ----- 사이드바: 추론 설정 -----
    with st.sidebar:
        st.header("⚙️ 탐지 설정")
        conf = st.slider("신뢰도 임계값 (conf)", 0.05, 0.95, DEFAULT_CONF, 0.05)
        iou = st.slider("NMS IoU 임계값", 0.1, 0.9, DEFAULT_IOU, 0.05)
        st.caption(f"탐지 가능 클래스 수: {len(model.names)}종")

    # ----- 1) 이미지 입력 -----
    st.subheader("1) 이미지 입력")
    tab_upload, tab_camera = st.tabs(["📁 파일 업로드", "📷 카메라 촬영"])
    with tab_upload:
        uploaded = st.file_uploader(
            "알약 사진을 업로드하세요 (jpg/png)",
            type=["jpg", "jpeg", "png"],
        )
    with tab_camera:
        captured = st.camera_input("알약을 촬영하세요")

    # 입력 소스 결정 (업로드 우선, 없으면 카메라)
    source = uploaded or captured
    if source is None:
        st.info("사진을 업로드하거나 촬영하면 탐지가 시작됩니다.")
        return

    pil_img = Image.open(io.BytesIO(source.getvalue()))

    # ----- 추론 실행 -----
    with st.spinner("탐지 중..."):
        result_img, detections = run_detection(model, pil_img, conf, iou)

    # ----- 2) 탐지 결과 이미지 -----
    st.subheader("2) 탐지 결과")
    col_in, col_out = st.columns(2)
    with col_in:
        st.image(pil_img, caption="원본 이미지", use_container_width=True)
    with col_out:
        st.image(result_img, caption=f"탐지 결과 ({len(detections)}개)", use_container_width=True)

    # ----- 3) 탐지 목록 -----
    st.subheader("3) 탐지 목록")
    if not detections:
        st.warning("탐지된 알약이 없습니다. 신뢰도 임계값을 낮추거나 다른 사진을 시도해 보세요.")
    else:
        table = [
            {
                "순번": i + 1,
                "품목명": d["name"],
                "신뢰도": f"{d['conf']*100:.1f}%",
                "위치 (x1,y1,x2,y2)": ", ".join(map(str, d["xyxy"])),
            }
            for i, d in enumerate(detections)
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

        # 결과 이미지 다운로드 버튼
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ 결과 이미지 저장",
            data=buf.getvalue(),
            file_name=f"pill_detection_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
