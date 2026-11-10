import os  # (추가) generate_img_summaries 에서 os.listdir 사용

# ⌨️ 2-1. StrOutputParser, ChatPromptTemplate, ChatOpenAI 를 임포트합니다.
#        (langchain_core.output_parsers / langchain_core.prompts / langchain_openai)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ⌨️ 6-1. base64 모듈과 HumanMessage(langchain_core.messages)를 임포트합니다.
import base64
from langchain_core.messages import HumanMessage


# ── 02-[9] ──────────────────────────────────────────────
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 요약본을 생성합니다.

    texts          : 텍스트 리스트
    tables         : 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부
    """

    # Prompt 한국어 버전
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

    # Prompt 영어 버전 (한국어 결과가 부정확할 때 교체)
    # prompt_text_eng = """You are an assistant tasked with summarizing tables and text for retrieval.
    # These summaries will be embedded and used to retrieve the raw text or table elements.
    # Give a concise summary of the table or text that is well optimized for retrieval.
    # Table or text: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)

    # 모델: gpt-4o-mini (교재의 gpt-4 대신 저렴하고 빠른 최신 모델)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 요약 체인 = 입력 포장 + 프롬프트 + LLM + 출력 파서
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트 요약을 활성화한 경우
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # 텍스트 요약을 사용하지 않는 경우 원본을 그대로 사용
    elif texts:
        text_summaries = texts

    # 표 데이터 요약
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


# ── 02-[19] ─────────────────────────────────────────────
# ⌨️ 6-2. encode_image(image_path) 함수를 정의합니다.
#        파일을 "rb" 로 열어 → b64encode → "utf-8" 로 decode 한 문자열을 반환
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ── 02-[22] ─────────────────────────────────────────────
# ⌨️ 7-1. image_summarize(img_base64, prompt) 함수를 정의합니다.
def image_summarize(img_base64, prompt):

#   ⌨️ (1) ChatOpenAI 로 chat 을 만듭니다. (model="gpt-4o-mini", max_tokens=1024)
    chat = ChatOpenAI(model="gpt-4o-mini"  , max_tokens=1024)
#   ⌨️ (2) 위 구조대로 HumanMessage 를 만들어 chat.invoke([...]) 로 호출합니다.
#          이미지 url 은 "data:image/jpeg;base64," 접두사 뒤에 img_base64 를 붙입니다.

    msg = chat.invoke([

        HumanMessage(content=[
                      {"type": "text",      "text": prompt} ,   # 수정: "지시문" 글자가 아니라 받은 prompt 변수를 보내야 함
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        },
      ])

    ])
    return msg.content

#   ⌨️ (3) 응답의 .content 를 반환합니다.


# ── 02-[25] ─────────────────────────────────────────────
# ⌨️ 8-1. generate_img_summaries(path) 함수를 정의합니다.

#   ⌨️ (1) img_base64_list, image_summaries 빈 리스트를 만듭니다.

#   ⌨️ (2) 7번과 같은 검색용 이미지 요약 프롬프트(prompt_kor)를 작성합니다.

#   ⌨️ (3) os.listdir(path) 를 정렬해 반복하면서
#          확장자가 (".jpg", ".jpeg", ".png") 인 파일만
#          → 전체 경로 만들기 → encode_image → 리스트에 추가
#          → image_summarize 결과를 요약 리스트에 추가

#   ⌨️ (4) img_base64_list, image_summaries 를 반환합니다.
def generate_img_summaries(path):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성

    path: Unstructured에 의해 추출된 이미지 파일들의 경로
    """
    img_base64_list = []   # Base64로 인코딩된 이미지
    image_summaries = []   # 이미지 요약문

    # 보강한 프롬프트: 제목·수치·예상 질문·키워드를 넣어야 질문과 같은 단어로 검색됨
    # (예: 제목이 잘린 '산업별 취업자수' 그래프가 '만명'으로만 요약돼 검색에서 빠졌던 문제)
    prompt_kor = """당신은 검색용 이미지 요약 도우미입니다. 이 요약은 임베딩되어 원본 이미지를 찾는 데 쓰입니다.
다음 항목을 빠짐없이 한국어로 적으세요.
1. 그래프 제목 (잘려서 안 보이면 축 단위와 항목을 보고 추정해서 '추정 제목:'으로 적기)
2. 그래프 종류, 범례, 축 단위
3. 모든 항목 이름과 수치 (숫자는 그림 그대로)
4. 핵심 추세 한 문장
5. 이 그래프로 답할 수 있는 질문 2개
6. 검색 키워드 5개 (예: 취업자수, 고용, 산업별)
그래프가 아닌 로고·표지·글자 이미지라면 무엇인지 한 줄로만 적으세요.
인사말이나 '도움이 필요하면 말씀해 주세요' 같은 문장은 쓰지 마세요."""

    # 예전 프롬프트 (비교용)
    # prompt_old = """You are an assistant tasked with summarizing images for retrieval.
    # These summaries will be embedded and used to retrieve the raw image.
    # Provide a concise summary of the image that is well optimized for retrieval.
    # The summary should be written in Korean (Hangul)."""

    for img_file in sorted(os.listdir(path)):
        # .jpg, .png, .jpeg 확장자만 처리합니다
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kor))

    return img_base64_list, image_summaries