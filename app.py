import hashlib

import streamlit as st

from face_reader import FORTUNE_KEYS, analyze_face, get_secret, result_to_text, validate_image

MAX_FILE_MB = 10
UPLOAD, CAMERA = "📁 사진 업로드", "📷 카메라 촬영"

st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")

st.title("🔮 AI 관상가")
st.caption("사진 한 장으로 보는 나의 관상")
st.info(
    "재미로 보는 결과이며 과학적 근거는 없어요. 사진은 저장하지 않고 분석에만 사용합니다.",
    icon="ℹ️",
)

if not get_secret("OPENAI_API_KEY"):
    st.error("OpenAI API 키가 설정되지 않았습니다.", icon="🔑")
    st.markdown(
        """
- **로컬 실행**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...`를 추가하세요.
- **Streamlit Cloud**: 앱 설정 → **Secrets**에 `OPENAI_API_KEY = "sk-..."`를 추가하세요.
"""
    )
    st.stop()


def star_markdown(score: int) -> str:
    filled = "★" * score
    empty = "☆" * (5 - score)
    return f":orange[{filled}]{empty}"


def render_result(data: dict) -> None:
    if not data["valid_face"]:
        st.warning(data["error_message"], icon="🙅")
        return

    st.divider()

    st.subheader("✨ 종합 총평")
    with st.container(border=True):
        st.markdown(f"#### “{data['summary']['headline']}”")
        if data["summary"]["description"]:
            st.write(data["summary"]["description"])
        if data["keywords"]:
            st.markdown(" ".join(f":violet-badge[{kw}]" for kw in data["keywords"]))

    st.subheader("🍀 운세")
    # 카드 4개는 모바일에서 세로로 길게 쌓이므로 카드 하나에 한 줄씩 정리한다.
    with st.container(border=True):
        for key, (icon, label) in FORTUNE_KEYS.items():
            item = data["fortune"][key]
            line = f"**{icon} {label}** &nbsp; {star_markdown(item['score'])}"
            if item["comment"]:
                line += f"  \n:gray[{item['comment']}]"
            st.markdown(line)

    st.subheader("👤 부위별 관상")
    features = data["features"]
    for i in range(0, len(features), 2):
        for col, feature in zip(st.columns(2), features[i : i + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{feature['part']}**")
                st.caption(feature["observation"])
                st.write(feature["interpretation"])

    if data["advice"]:
        st.subheader("💡 조언")
        st.success(data["advice"], icon="🌟")

    st.download_button(
        "📄 결과 텍스트로 저장",
        data=result_to_text(data),
        file_name="관상_결과.txt",
        mime="text/plain",
        width="stretch",
    )


# 카메라 위젯은 선택했을 때만 그린다. 탭에 넣으면 숨겨져 있어도 페이지를 열자마자 카메라 권한을 요청한다.
# required=True: 선택된 버튼을 다시 눌러도 선택이 풀리지 않게 한다.
mode = st.segmented_control(
    "사진 입력 방식", [UPLOAD, CAMERA], default=UPLOAD, required=True, label_visibility="collapsed"
)
if mode == CAMERA:
    image = st.camera_input("얼굴이 정면으로 잘 보이게 촬영해주세요")
else:
    image = st.file_uploader(
        "얼굴 사진을 올려주세요",
        type=["jpg", "jpeg", "png", "webp"],
        help="한 사람의 얼굴이 정면으로 잘 보이는 사진이 가장 정확해요.",
    )
    # 10MB 초과 파일은 업로드 위젯이 먼저 막고 영어 툴팁만 띄우므로 한국어로 미리 안내한다.
    st.caption(f"JPG·PNG·WEBP, {MAX_FILE_MB}MB 이하. 파일 옆에 빨간 표시가 뜨면 용량을 줄여 다시 올려주세요.")

if image is None:
    st.session_state.pop("analysis", None)
    st.stop()

image_bytes = image.getvalue()
if len(image_bytes) > MAX_FILE_MB * 1024 * 1024:
    st.session_state.pop("analysis", None)
    st.error(f"파일이 너무 큽니다. {MAX_FILE_MB}MB 이하의 사진을 올려주세요.", icon="📦")
    st.stop()

# 미리보기(st.image)도 이미지를 디코딩하므로, 깨진 파일이나 픽셀 폭탄은 그리기 전에 막는다.
invalid = validate_image(image_bytes)
if invalid:
    st.session_state.pop("analysis", None)
    st.error(invalid, icon="🖼️")
    st.stop()

image_hash = hashlib.sha256(image_bytes).hexdigest()
cached = st.session_state.get("analysis")
if cached and cached["hash"] != image_hash:
    # 다른 사진으로 바뀌면 이전 결과를 지운다.
    st.session_state.pop("analysis", None)
    cached = None

# 카메라 모드에서는 위젯이 이미 사진을 보여주므로 미리보기를 생략한다.
if mode != CAMERA:
    # 고정 폭: 모바일에서 사진이 화면을 다 차지해 버튼이 아래로 밀리지 않게 한다.
    st.image(image_bytes, caption="업로드한 사진", width=240)

if st.button("🔍 관상 보기", type="primary", width="stretch") and cached is None:
    with st.spinner("관상을 보는 중입니다... (최대 30초 정도 걸려요)"):
        result = analyze_face(image_bytes)
    cached = {"hash": image_hash, "result": result}
    # 오류와 "분석 불가" 응답은 다시 시도할 수 있도록 캐시하지 않는다.
    # (모델이 거절을 valid_face=false로 돌려주는 경우가 있어, 캐시하면 재시도가 막힌다.)
    if result.error is None and result.data is not None and result.data["valid_face"]:
        st.session_state["analysis"] = cached

if cached:
    result = cached["result"]
    if result.error:
        st.error(result.error, icon="⚠️")
    elif result.data is not None:
        render_result(result.data)
    else:
        st.warning("결과를 정리된 형태로 읽지 못해 원문을 그대로 보여드립니다.", icon="📝")
        st.text(result.raw or "(빈 응답)")
