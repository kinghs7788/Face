"""OpenAI Vision 호출로 얼굴 사진의 관상을 분석한다."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

DEFAULT_MODEL = "gpt-4o"
MAX_SIDE = 1024
# 파일은 작아도 픽셀 수가 큰 이미지(픽셀 폭탄)는 열 때 메모리를 크게 쓰므로 미리 거른다.
MAX_PIXELS = 50_000_000
IMAGE_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError)
MAX_COMPLETION_TOKENS = 2000

FEATURE_PARTS = ["얼굴형", "이마", "눈썹", "눈", "코", "입", "귀", "턱"]
FORTUNE_KEYS = {
    "wealth": ("💰", "재물운"),
    "love": ("❤️", "애정운"),
    "career": ("💼", "직업운"),
    "health": ("💪", "건강운"),
}

# "누구인지 언급하지 말라"를 금지 조항으로만 쓰면 gpt-4o가 이를 근거로 분석 자체를 거절했다(실측 3/3 거절).
# 신원 확인이 아니라 보이는 이목구비만 묘사하는 작업임을 먼저 밝히고, valid_face=false 조건을 세 가지로 못박는다.
SYSTEM_PROMPT = f"""너는 전통 관상학(오관, 삼정, 십이궁 등)을 재미있게 풀어주는 엔터테인먼트 콘텐츠 작가다.
이 앱의 사용자는 자기 사진(또는 동의를 받은 지인의 사진)을 직접 올려 재미로 관상 풀이를 받는다.

## 이 작업이 무엇인지
- 이 작업은 신원 확인이 아니다. 사진 속 사람이 누구인지 알 필요가 전혀 없고, 이름이나 닮은 사람을 언급하지 않으면 된다.
- 눈에 보이는 이목구비의 모양(예: 이마가 넓다, 눈꼬리가 살짝 올라갔다, 입꼬리가 올라가 있다)만 묘사하고, 그것을 전통 관상학의 상징으로 재미있게 풀이한다.
- 누구인지 모른다는 것은 거절 사유가 아니다. 얼굴이 보이면 반드시 풀이한다.

## 원칙
- 모든 답변은 한국어로 작성한다.
- 오락용이며 과학적 근거가 없다는 전제로, 따뜻하고 긍정적이며 건설적인 톤을 유지한다.
- 약점을 말할 때도 보완할 방법과 함께 부드럽게 표현한다.
- 인종, 민족, 성별, 나이, 외모의 우열을 평가하거나 비하하는 표현을 쓰지 않는다.
- 건강운은 일반적인 생활 조언 수준으로만 쓰고 질병을 진단하지 않는다.

## valid_face를 false로 하는 경우 (이 세 가지뿐)
- 사진에 사람 얼굴이 없다.
- 얼굴이 두 명 이상이다.
- 얼굴이 너무 작거나, 흐리거나, 가려져서 이목구비 모양을 알아볼 수 없다.
이 경우 error_message에 사용자가 이해할 수 있는 한국어 안내를 쓰고 나머지 필드는 비워둔다.

## 출력 형식
반드시 아래 구조의 JSON 객체 하나만 출력한다.
{{
  "valid_face": true,
  "error_message": "",
  "summary": {{
    "headline": "관상을 한 문장으로 요약 (20자 내외)",
    "description": "종합 총평 3~5문장"
  }},
  "features": [
    {{"part": "얼굴형", "observation": "사진에서 관찰된 특징", "interpretation": "관상학적 해석 2~3문장"}}
  ],
  "fortune": {{
    "wealth": {{"score": 1~5 정수, "comment": "한 줄 설명"}},
    "love": {{"score": 1~5 정수, "comment": "한 줄 설명"}},
    "career": {{"score": 1~5 정수, "comment": "한 줄 설명"}},
    "health": {{"score": 1~5 정수, "comment": "한 줄 설명"}}
  }},
  "keywords": ["#키워드1", "#키워드2", "#키워드3"],
  "advice": "운을 좋게 하는 조언 한마디"
}}

- features는 정확히 다음 8개 부위를 이 순서대로 포함한다: {", ".join(FEATURE_PARTS)}
- 귀처럼 사진에서 보이지 않는 부위는 observation에 "사진에서 잘 보이지 않음"이라고 쓰고, 보이는 다른 특징을 바탕으로 짧게 해석한다.
- keywords는 '#'으로 시작하는 3~5개 항목이다.
"""


@dataclass
class AnalysisResult:
    """분석 결과. data가 있으면 정상, error가 있으면 오류, raw만 있으면 JSON 파싱 실패."""

    data: dict | None = None
    raw: str | None = None
    error: str | None = None


def get_secret(key: str, default: str | None = None) -> str | None:
    """Streamlit Cloud의 st.secrets를 우선 읽고, 없으면 환경변수(.env)를 읽는다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # 로컬에 .streamlit/secrets.toml이 없으면 st.secrets 접근 시 예외가 난다.
        pass
    return os.getenv(key, default)


def validate_image(image_bytes: bytes) -> str | None:
    """화면에 그리거나 분석하기 전에 이미지를 검사한다. 문제가 있으면 사용자용 메시지를 돌려준다."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size  # 헤더만 읽으므로 픽셀 폭탄도 메모리를 쓰지 않는다.
            if width * height > MAX_PIXELS:
                return f"사진 해상도가 너무 큽니다({width}×{height}). 크기를 줄여서 다시 올려주세요."
            # verify()는 잘린 JPEG를 통과시키므로 실제로 디코딩해 본다.
            # JPEG는 draft로 축소 디코딩해 메모리를 아낀다(다른 형식은 무시됨).
            img.draft("RGB", (MAX_SIDE, MAX_SIDE))
            img.load()
    except Image.DecompressionBombError:
        return "사진 해상도가 너무 큽니다. 크기를 줄여서 다시 올려주세요."
    except IMAGE_ERRORS:
        return "이미지 파일을 읽을 수 없습니다. 손상되지 않은 JPG, PNG, WEBP 사진인지 확인해주세요."
    return None


def preprocess_image(image_bytes: bytes) -> tuple[bytes, str]:
    """EXIF 회전 보정, RGB 변환, 긴 변 MAX_SIDE 리사이즈 후 JPEG 바이트를 돌려준다."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), "image/jpeg"


def _to_score(value) -> int:
    try:
        return min(5, max(1, int(round(float(value)))))
    except (TypeError, ValueError):
        return 3


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _to_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("false", "no", "0", "아니오"):
            return False
        if lowered in ("true", "yes", "1", "예"):
            return True
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _part_key(name: str) -> str:
    return "".join(name.split())


def normalize_result(data: dict) -> dict:
    """모델 응답의 누락 키나 잘못된 타입을 보정해 화면에서 KeyError가 나지 않게 한다."""
    if not _to_bool(data.get("valid_face"), default=True):
        return {
            "valid_face": False,
            "error_message": _text(data.get("error_message"))
            or "얼굴을 찾지 못했습니다. 한 사람의 얼굴이 잘 보이는 사진으로 다시 시도해주세요.",
        }

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}

    raw_features = data.get("features") if isinstance(data.get("features"), list) else []
    raw_features = [item for item in raw_features if isinstance(item, dict)]
    # "얼굴 형", "눈 "처럼 공백이 섞여도 매칭되도록 공백을 지운 이름으로 비교한다.
    features_by_part = {}
    for item in raw_features:
        features_by_part.setdefault(_part_key(_text(item.get("part"))), item)
    # 부위 이름이 하나도 맞지 않으면(예: 영어로 답함) 프롬프트에서 지정한 순서대로 대응시킨다.
    use_order = not any(_part_key(p) in features_by_part for p in FEATURE_PARTS)
    features = []
    for index, part in enumerate(FEATURE_PARTS):
        if use_order:
            item = raw_features[index] if index < len(raw_features) else {}
        else:
            item = features_by_part.get(_part_key(part), {})
        features.append(
            {
                "part": part,
                "observation": _text(item.get("observation")) or "정보 없음",
                "interpretation": _text(item.get("interpretation")) or "해석 정보가 없습니다.",
            }
        )

    raw_fortune = data.get("fortune") if isinstance(data.get("fortune"), dict) else {}
    fortune = {}
    for key in FORTUNE_KEYS:
        item = raw_fortune.get(key) if isinstance(raw_fortune.get(key), dict) else {}
        fortune[key] = {"score": _to_score(item.get("score")), "comment": _text(item.get("comment"))}

    raw_keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    keywords = []
    for kw in raw_keywords:
        kw = _text(kw)
        if kw:
            keywords.append(kw if kw.startswith("#") else f"#{kw}")

    return {
        "valid_face": True,
        "error_message": "",
        "summary": {
            "headline": _text(summary.get("headline")) or "관상 풀이 결과",
            "description": _text(summary.get("description")),
        },
        "features": features,
        "fortune": fortune,
        "keywords": keywords[:5],
        "advice": _text(data.get("advice")),
    }


def analyze_face(image_bytes: bytes) -> AnalysisResult:
    """이미지를 OpenAI 비전 모델에 보내고 정규화된 결과를 돌려준다."""
    invalid = validate_image(image_bytes)
    if invalid:
        return AnalysisResult(error=invalid)
    try:
        prepared, mime = preprocess_image(image_bytes)
    except IMAGE_ERRORS:
        return AnalysisResult(error="이미지 파일을 읽을 수 없습니다. 손상되지 않은 JPG, PNG, WEBP 사진인지 확인해주세요.")

    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"), timeout=60)
    model = get_secret("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    b64 = base64.b64encode(prepared).decode()

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "제 사진입니다. 재미로 관상 풀이를 해주세요. 보이는 이목구비를 바탕으로 정해진 JSON 형식으로 답해주세요."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except openai.AuthenticationError:
        return AnalysisResult(error="OpenAI API 키가 올바르지 않습니다. 키 설정을 확인해주세요.")
    except openai.PermissionDeniedError:
        return AnalysisResult(error="이 API 키로는 해당 모델을 사용할 권한이 없습니다.")
    except openai.NotFoundError:
        return AnalysisResult(error=f"'{model}' 모델을 찾을 수 없습니다. OPENAI_MODEL 설정을 확인해주세요.")
    except openai.RateLimitError:
        return AnalysisResult(error="요청이 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도하거나 OpenAI 결제 한도를 확인해주세요.")
    except openai.APITimeoutError:
        return AnalysisResult(error="분석 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
    except openai.APIConnectionError:
        return AnalysisResult(error="OpenAI 서버에 연결하지 못했습니다. 네트워크 상태를 확인해주세요.")
    except openai.BadRequestError:
        return AnalysisResult(error="요청을 처리하지 못했습니다. 다른 사진으로 다시 시도해주세요.")
    except openai.OpenAIError:
        return AnalysisResult(error="OpenAI 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    choice = response.choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        return AnalysisResult(error="AI가 이 사진의 분석을 거절했습니다. 한 사람의 얼굴이 정면으로 잘 보이는 다른 사진으로 시도해주세요.")

    content = message.content or ""
    if choice.finish_reason == "length":
        return AnalysisResult(raw=content, error="응답이 너무 길어 중간에 잘렸습니다. 다시 시도해주세요.")

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("JSON 객체가 아님")
    except ValueError:
        return AnalysisResult(raw=content)

    return AnalysisResult(data=normalize_result(data))


def stars(score: int) -> str:
    return "★" * score + "☆" * (5 - score)


def result_to_text(data: dict) -> str:
    """다운로드용 텍스트를 만든다."""
    lines = ["[AI 관상 결과]", "", f"■ 종합 총평: {data['summary']['headline']}", data["summary"]["description"], ""]
    lines.append("■ 운세")
    for key, (_, label) in FORTUNE_KEYS.items():
        item = data["fortune"][key]
        lines.append(f"- {label} {stars(item['score'])} {item['comment']}")
    lines += ["", "■ 부위별 관상"]
    for feature in data["features"]:
        lines.append(f"[{feature['part']}]")
        lines.append(f"  특징: {feature['observation']}")
        lines.append(f"  해석: {feature['interpretation']}")
    if data["keywords"]:
        lines += ["", "■ 키워드", " ".join(data["keywords"])]
    if data["advice"]:
        lines += ["", "■ 조언", data["advice"]]
    lines += ["", "※ 재미로 보는 결과이며 과학적 근거가 없습니다."]
    return "\n".join(lines)
