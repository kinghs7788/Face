# 🔮 AI 관상가

얼굴 사진을 올리면 OpenAI 비전 모델이 전통 관상학 관점에서 풀이해 주는 Streamlit 앱입니다.
재미로 보는 결과이며 과학적 근거는 없습니다. 업로드한 사진은 저장하지 않습니다.

## 기능

- 사진 업로드(JPG·PNG·WEBP, 10MB 이하) 또는 카메라 촬영
- 종합 총평, 운세 4종(재물·애정·직업·건강) 별점, 부위별 관상 8개, 키워드, 조언
- 결과 텍스트 파일 저장
- 얼굴이 없거나 여러 명인 사진은 분석하지 않고 안내

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
streamlit run app.py
```

| 환경변수 | 설명 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | |
| `OPENAI_MODEL` | 이미지 입력을 지원하는 모델 | `gpt-4o` |

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 push합니다. `.env`는 `.gitignore`에 있어 올라가지 않습니다.
2. [share.streamlit.io](https://share.streamlit.io)에서 **New app** → 저장소, 브랜치, `app.py`를 선택합니다.
3. **Advanced settings → Secrets**에 키를 입력하고 배포합니다.

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o"
```

Streamlit Cloud는 `.env`를 읽지 않으므로 Secrets 설정이 필요합니다. 공개 URL로 배포하면 누구나 API를 호출할 수 있으니 OpenAI 대시보드에서 월 사용 한도를 설정해 두세요.

## 테스트 이미지

```bash
python tests/make_fixtures.py
```

`tests/fixtures/`에 풍경, 투명 PNG, 10MB 초과, 깨진 파일 등 예외 상황용 이미지를 만듭니다. 실제 얼굴 사진(`face.jpg`)은 직접 넣어야 하며, 이 폴더는 Git에 올라가지 않습니다.

## 구조

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 화면, 입력 검사, 결과 캐시와 렌더링 |
| `face_reader.py` | 이미지 검사·전처리, OpenAI 호출, 응답 정규화 |
| `.streamlit/config.toml` | 라이트/다크 테마 색, 업로드 용량 제한 |
| `PRD.md` | 요구사항 명세 |
