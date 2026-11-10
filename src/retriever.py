import re # 정규표현식으로 문자열 패턴, 만든 검사, 치환, 추출 등 다양한 문자열 처리 기능
import uuid # 고유한 id 생성 (체인 답변 생성 시 id 부여) 요약본, 원본을 doc_id로 페어링

# 1) langchain_core.documents 의 Document import
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 6) 연결 키(= 청구기호) : id_key = "doc_id"
# (추가) 03-[6] 에 있던 값. add_documents 가 사용하므로 이 파일로 옮김
id_key = "doc_id"


# ── 03-[8] ──────────────────────────────────────────────
# 2) add_documents(retriever, doc_summaries, doc_contents) 함수 정의
#    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
#    - 원본 개수만큼 uuid4 로 doc_ids 생성 (문자열로 변환)
#    - 요약문마다 Document(page_content=요약, metadata={id_key: doc_ids[i]}) 생성
def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = []
    doc_ids =[str(uuid.uuid4())for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i,s in enumerate(doc_summaries)
    ]
#    - TODO 1: 요약본은 어디에 넣을까요? (vectorstore / docstore)
#      → retriever.vectorstore.add_documents(...)
    retriever.vectorstore.add_documents(summary_docs)
#    - TODO 2: 원본은 어디에 넣을까요?
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


# ── 03-[13] ─────────────────────────────────────────────
# 2) split_image_text_types(docs) 함수 정의
#    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
#    - b64_images, texts_out 빈 리스트 준비
#    - Document 면 page_content 를 꺼내 문자열로 만들기
#    - Base64 문자열은 영문/숫자/+/= 만으로 길게 이어집니다
#      → 정규식 "^[A-Za-z0-9+/]{100,}={0,2}$" 매칭 시 이미지, 아니면 텍스트
#    - {"images": b64_images, "texts": texts_out} 반환
def split_image_text_types(docs):
    b64_images = []
    texts_out = []

    for doc in  docs:
           doc = doc.page_content if isinstance(doc, Document) else doc
           if re.match("^[A-Za-z0-9+/]{100,}={0,2}$",doc):
                 b64_images.append(doc)
           else:
                 texts_out.append(doc)

    return {"images" : b64_images, "texts": texts_out}


# ── 03-[15] ─────────────────────────────────────────────
# 2) img_prompt_func(data_dict) 함수 정의
#    """검색된 원본을 멀티모달 LLM이 읽을 형태로 조립합니다."""
#    - messages 빈 리스트 준비
#    - 1) 이미지를 먼저 넣습니다 (모델이 그림을 먼저 보게)
#         data_dict["context"]["images"] 마다 {"type": "image_url", "image_url": {"url": ...}} 추가
#         TODO 4: Base64 앞에 붙일 접두사를 완성하세요 (data:image/jpeg;base64,)
#    - 2) 텍스트 컨텍스트와 질문을 넣습니다
#         texts 를 줄바꿈으로 합치고 {"type": "text", "text": ...} 추가
#         지시문: "아래 자료를 근거로 질문에 한국어로 답하세요. 자료에 없는 내용은 모른다고 답하세요."
#         + 질문(data_dict["question"]) + 참고 텍스트
#    - [HumanMessage(content=messages)] 반환
def img_prompt_func(data_dict):
    messages = []
    # 1) 이미지를 먼저 넣습니다 (모델이 그림을 먼저 보게)
    for img in data_dict["context"]["images"]:
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    # 2) 텍스트 컨텍스트와 질문을 넣습니다 (반복문 밖!)
    texts = "\n".join(data_dict["context"]["texts"])
    messages.append({"type": "text", "text": "아래 자료를 근거로 질문에 한국어로 답하세요. "
                     "그래프로 나타난 수치는 이미지를 분석해서 값을 요약에 포함하세요. "
                     "자료에 없는 내용은 모른다고 답하세요.\n"
                     f"질문: {data_dict['question']}\n참고 자료:\n{texts}"})
    return [HumanMessage(content=messages)]