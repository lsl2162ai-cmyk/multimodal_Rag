# 🍎 멀티모달 RAG, 초등학생도 이해하는 정리 노트

> 지금까지 배운 3가지 실습을 한 번에 정리했어요.
> **01 데이터 추출 → 02 요약 생성 → 03 임베딩 + 최종 체인**

---

## 0. 먼저 큰 그림부터! 🏫 "똑똑한 도서관 사서" 이야기

여러분 학교 도서관에 **아주 두꺼운 책(PDF)** 이 한 권 있어요.
그 책 안에는 **글**도 있고, **표**도 있고, **그림(그래프)** 도 있어요.

친구가 사서 선생님께 이렇게 물어봐요.

> "선생님, 그래프에 나온 경제 추세가 어때요?"

똑똑한 사서 선생님은 이렇게 일해요.

| 순서 | 사서 선생님이 하는 일 | 우리가 배운 실습 |
| --- | --- | --- |
| ① | 책을 **글 / 표 / 그림**으로 가위로 오려서 나눠요 | **01 데이터 추출** |
| ② | 오린 조각마다 **"이건 이런 내용이야"** 라고 짧은 메모(색인 카드)를 써요 | **02 요약 생성** |
| ③ | 메모는 **카드 서랍**에, 진짜 조각은 **책장**에 넣고 **같은 번호표**를 붙여요 | **03 임베딩** |
| ④ | 질문이 오면 **메모로 찾고 → 번호표로 진짜 조각을 꺼내 → 보고 대답**해요 | **03 최종 체인** |

```
PDF ──▶ ① 오리기 ──▶ ② 메모 쓰기 ──▶ ③ 서랍+책장에 정리 ──▶ ④ 질문에 답하기
       (추출)        (요약)          (임베딩/저장)           (체인)
```

💡 **멀티모달**이란? → "여러 가지 모양(글, 그림, 표)을 함께 다룬다"는 뜻이에요.
💡 **RAG**란? → "먼저 **찾아보고(Retrieval)**, 그걸 보고 **대답을 만든다(Generation)**"는 뜻이에요. 시험 볼 때 교과서를 펴 보고 답을 쓰는 것과 같아요!

---

## 1교시 📄✂️ 데이터 추출 (01_데이터추출)

### 쉽게 말하면
**PDF라는 큰 책을 글 / 표 / 그림으로 분리수거하는 단계**예요.
그리고 글이 너무 길면 **적당한 크기로 잘라요.** (피자를 한 조각씩 자르는 것처럼 🍕)

### 핵심 코드 ① — PDF를 자동 분리수거 기계에 넣기

```python
def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,               # 어떤 PDF를 읽을까?
        strategy="hi_res",                   # 꼼꼼하게(고화질로) 읽어라
        extract_images_in_pdf=True,          # 그림도 꺼내라
        infer_table_structure=True,          # 표는 칸 모양을 살려라
        chunking_strategy="by_title",        # 제목 기준으로 묶어라
        max_characters=4000,                 # 한 덩어리는 최대 4000자
        new_after_n_chars=3800,              # 3800자 넘으면 새 덩어리
        combine_text_under_n_chars=2000,     # 2000자보다 작으면 합쳐라
        extract_image_block_output_dir=path + "/figures",  # 그림 저장 폴더
    )
```

**🔍 코드 읽는 법**
- `def 이름(재료):` → "이런 일을 하는 **기계를 만든다**"는 뜻. 재료는 `path`(폴더), `fname`(파일 이름).
- `return` → 기계가 일한 **결과물을 돌려준다**.
- `partition_pdf(...)` → 남이 만들어 둔 **분리수거 기계**. 괄호 안은 기계의 **설정 버튼**이에요.
- `strategy="hi_res"` → 돋보기 모드! 페이지를 사진처럼 보고 표·그림·제목을 찾아요. 대신 **느려요** (2~3분).
  - `"fast"` 로 하면 빠르지만 글자만 읽어서 표·그림을 놓쳐요.

### 핵심 코드 ② — 분리된 조각을 "글 바구니"와 "표 바구니"에 담기

```python
def categorize_elements(raw_pdf_elements):
    tables = []      # 표 바구니 (빈 상자)
    texts = []       # 글 바구니 (빈 상자)

    for element in raw_pdf_elements:              # 조각을 하나씩 꺼내서
        if "...Table" in str(type(element)):      # 이게 표라면
            tables.append(str(element))           #   → 표 바구니에 넣고
        elif "...CompositeElement" in str(type(element)):  # 글 덩어리라면
            texts.append(str(element))            #   → 글 바구니에 넣어요

    return texts, tables                          # 바구니 2개를 돌려줘요
```

**🔍 코드 읽는 법**
- `[]` → **빈 상자(리스트)**. `.append(x)` → 상자에 `x`를 **하나 넣기**.
- `for 하나 in 여러개:` → "여러 개를 **하나씩 차례로** 꺼내서 아래 일을 해라" (반복).
- `if / elif` → "만약 ~라면 / 아니고 만약 ~라면" (조건).
- `type(element)` → "너는 **무슨 종류**니?" 하고 물어보는 것.

### 핵심 코드 ③ — 긴 글을 피자처럼 자르기 (겹치게!)

```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,      # 한 조각 크기 = 2000 토큰
    chunk_overlap=200,    # 앞 조각과 200만큼 겹치게
)
joined_texts = " ".join(texts)                      # 글 조각들을 한 줄로 이어 붙이고
texts_2k_token = text_splitter.split_text(joined_texts)  # 다시 2000씩 잘라요
```

**🤔 왜 200만큼 겹치게 자를까?**
"나는 오늘 학교에 / 가서 친구를 만났다" 처럼 **문장 중간에서 잘리면 뜻이 사라져요.**
조금씩 겹치게 자르면 잘린 부분의 뜻이 **양쪽 조각에 다 남아요.** 🧩

### 핵심 코드 ④ — 결과를 저장해 두기 (pickle = 냉동실 🧊)

```python
with open(output_dir / "extracted.pkl", "wb") as f:
    pickle.dump({"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token}, f)
```

- 추출은 2~3분이나 걸리니까 **다음 시간에 또 하지 않으려고 냉동 보관**하는 거예요.
- `"wb"` = **w**rite(쓰기) + **b**inary(컴퓨터 언어로). 꺼낼 땐 `"rb"` = **r**ead(읽기).
- `with open(...) as f:` → "파일을 열고, 일이 끝나면 **자동으로 닫아줘**".

---

## 2교시 📝 요약 생성 (02_요약생성)

### 쉽게 말하면
오려낸 조각마다 **AI에게 "짧게 요약해 줘"** 라고 부탁해서 **색인 카드(메모)** 를 만드는 단계예요.
- 글·표 → 글을 읽는 AI(**LLM**)가 요약
- 그림 → **눈이 달린 AI(VLM)** 가 그림을 보고 설명

⚠️ 이번 시간부터 **AI를 부를 때마다 돈(API 비용)이 나가요!** 같은 셀을 여러 번 돌리지 않기.

### 핵심 코드 ① — 비밀 열쇠(API 키) 꺼내기 🔑

```python
load_dotenv(PROJECT_ROOT / ".env")           # .env 비밀 쪽지를 읽고
openai_api_key = os.getenv("OPENAI_API_KEY") # 그 안의 열쇠를 꺼내요
if not openai_api_key:
    raise ValueError("OpenAI API 키가 없습니다.")  # 열쇠가 없으면 멈춰!
```

- `raise` → "**문제 발생! 여기서 멈춰!**" 하고 소리치는 것.
- 열쇠 전체를 `print()` 하면 안 돼요. 남이 보면 내 돈으로 AI를 쓸 수 있어요!

### 핵심 코드 ② — 요약 컨베이어 벨트 (LCEL) 🏭 ★가장 중요★

```python
prompt = ChatPromptTemplate.from_template("...요약할 표 또는 텍스트: {element}")
model  = ChatOpenAI(temperature=0, model="gpt-4o-mini")

summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
```

**🔍 코드 읽는 법 — `|` (파이프) 는 컨베이어 벨트!**

```
"원본 글"  →  {"element": "원본 글"}  →  "요약해줘: 원본 글"  →  AI의 답장  →  "요약문"
             ① 포장하기                 ② 편지지에 쓰기         ③ AI에게 보내기  ④ 내용만 꺼내기
```

| 칸 | 하는 일 | 비유 |
| --- | --- | --- |
| `{"element": lambda x: x}` | 받은 글을 `element`라는 이름표가 붙은 상자에 담기 | 📦 포장 |
| `prompt` | 편지지 양식의 `{element}` 빈칸에 글을 끼워 넣기 | ✉️ 편지 쓰기 |
| `model` | AI에게 편지 보내고 답장 받기 | 📮 우체국 |
| `StrOutputParser()` | 답장 봉투를 뜯고 **글자만** 꺼내기 | 📃 개봉 |

- `lambda x: x` → "받은 걸 **그대로** 돌려주는 아주 작은 기계". (이름 없는 한 줄짜리 함수)
- `temperature=0` → AI가 **장난치지 않고 매번 똑같이 성실하게** 대답하게 하는 설정.
- `.batch(여러개, {"max_concurrency": 5})` → 여러 개를 **5개씩 동시에** 처리. 너무 높이면 AI 회사가 "그만 좀 보내!" 하고 막아요 (`RateLimitError`).

### 핵심 코드 ③ — 그림을 글자로 포장하기 (Base64) 📦

```python
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
```

- AI에게는 **글자만** 보낼 수 있어요. 그래서 그림을 `iVBORw0KGgo...` 같은 **긴 글자 줄**로 바꿔서 보내요.
- 그림을 택배로 보내려고 **글자 상자에 포장**하는 것과 같아요.

### 핵심 코드 ④ — 눈 달린 AI에게 그림 보여주기 👀

```python
def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},                       # 부탁하는 말
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},  # 그림
        ])
    ])
    return msg.content
```

**🔍 코드 읽는 법**
- `HumanMessage(content=[글, 그림])` → **"말 + 사진"을 한 메시지에 같이 보내기**. 카톡에서 사진이랑 글을 같이 보내는 것!
- `f"...{img_base64}"` → **f-문자열**. `{ }` 안의 변수 값이 그 자리에 **쏙 들어가요**.
- `data:image/jpeg;base64,` → "지금부터 나오는 글자는 **jpeg 그림을 포장한 것**이야" 라는 **택배 송장**.
- `.invoke(...)` → AI를 **한 번 불러서** 일 시키기. (`.batch`는 여러 번)

### 좋은 요약 vs 나쁜 요약

| 수준 | 예시 | 나중에 찾을 수 있을까? |
| --- | --- | --- |
| 😢 나쁨 | "차트 이미지입니다" | 거의 못 찾아요 |
| 😐 보통 | "매출 관련 그래프" | 조금 |
| 😄 좋음 | "2021~2023년 분기별 매출 선그래프, 2023년 하반기 하락" | 잘 찾아요! |

→ 요약이 엉터리면 **검색 자체가 틀려요.** 그래서 그림과 요약을 **눈으로 꼭 비교**해야 해요.

---

## 3교시 🗄️ 임베딩 + 최종 체인 (03_임베딩_최종체인)

### 쉽게 말하면
1. 메모(요약)를 **숫자 좌표(임베딩)** 로 바꿔서 **카드 서랍(벡터 DB)** 에 넣어요.
2. 진짜 조각(원본)은 **책장(docstore)** 에 넣어요.
3. 둘에게 **같은 번호표(doc_id)** 를 붙여요. ← ★오늘의 한 줄 핵심★
4. 질문이 오면 → 서랍에서 비슷한 메모를 찾고 → 번호표로 책장의 원본을 꺼내 → AI가 보고 대답!

💡 **임베딩**이란? → 글의 **뜻을 숫자 좌표로 바꾸는 것**. 뜻이 비슷한 글은 지도에서 **가까운 곳**에 찍혀요.
"강아지"와 "멍멍이"는 글자는 달라도 좌표가 가까워요! 🐶📍

### 핵심 코드 ① — 서랍 + 책장 + 번호표 = 다중 벡터 검색기

```python
vectorstore = Chroma(collection_name="multimodal_rag",
                     embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
store  = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
```

| 코드 | 비유 | 안에 들어가는 것 | 역할 |
| --- | --- | --- | --- |
| `vectorstore` (Chroma) | 🗂️ 카드 서랍 | 요약문(숫자 좌표) | **찾기** |
| `store` (InMemoryStore) | 📚 책장 | 원본 글·표·그림(Base64) | **건네주기** |
| `id_key="doc_id"` | 🏷️ 번호표 | 같은 번호 | **둘을 연결** |
| `retriever` | 👩‍🏫 사서 선생님 | 위 세 개를 모두 가짐 | 찾고 꺼내기 |

⚠️ `InMemoryStore`는 **기억(메모리)에만** 저장해요. 컴퓨터(커널)를 끄면 **싹 잊어버려요.**

### 핵심 코드 ② — 같은 번호표 붙이기 🏷️

```python
def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]      # 원본 수만큼 번호표 만들기
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})  # 요약에 번호표 붙이기
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)          # 요약 → 서랍
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))  # 원본 → 책장 (같은 번호로)
```

**🔍 코드 읽는 법**
- `uuid.uuid4()` → 세상에 하나뿐인 **랜덤 번호표** (예: `3f2a9c...`).
- `[무엇 for _ in 여러개]` → **리스트 컴프리헨션**. "여러 개만큼 반복해서 **무엇을 만들어 상자에 담아라**"를 한 줄로!
  - `_` → "이 값은 안 쓸 거야"라는 표시.
- `enumerate(목록)` → 꺼낼 때 **순서 번호(i)도 같이** 줘요. (0번, 1번, 2번…)
- `zip(번호표들, 원본들)` → 지퍼처럼 **두 줄을 짝지어** `(번호표, 원본)` 쌍을 만들어요. 🤐
- `mset` → **m**ulti **set** = 여러 개를 한 번에 저장.
- ⚠️ 이 셀을 **두 번 돌리면 카드가 2장씩** 들어가요!

### 핵심 코드 ③ — 검색해 보기 🔎

```python
found = retriever.invoke("그래프에 나타난 추세를 설명해줘")
```

사서 선생님(retriever) 머릿속에서 일어나는 일:
1. 질문을 **숫자 좌표**로 바꿔요.
2. 서랍에서 **좌표가 가장 가까운 메모**를 찾아요.
3. 메모에 붙은 **번호표(doc_id)** 를 읽어요.
4. 책장에서 같은 번호의 **원본**을 꺼내 줘요.

👉 신기한 점: 질문과 **글자가 하나도 안 겹치는 그림**도 찾아와요! 메모(요약)를 거쳐서 찾았기 때문이에요.

### 핵심 코드 ④ — 꺼낸 원본을 "그림"과 "글"로 나누기

```python
def split_image_text_types(docs):
    b64_images, texts_out = [], []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$", doc):   # 생김새가 Base64면
            b64_images.append(doc)                          #   → 그림
        else:
            texts_out.append(doc)                           #   → 글
    return {"images": b64_images, "texts": texts_out}
```

**🔍 정규식 `^[A-Za-z0-9+/]{100,}={0,2}$` 읽는 법** (암호 해독! 🕵️)

| 조각 | 뜻 |
| --- | --- |
| `^` | 처음부터 |
| `[A-Za-z0-9+/]` | 영어 대·소문자, 숫자, `+`, `/` 중 하나가 |
| `{100,}` | **100개 이상** 이어지고 |
| `={0,2}` | 끝에 `=` 가 0~2개 붙고 |
| `$` | 거기서 끝나면 |

→ "띄어쓰기도 한글도 없이 영어·숫자만 엄청 길게 이어진 글자 = 포장된 그림!" 이라고 알아보는 거예요.

### 핵심 코드 ⑤ — 최종 체인 (컨베이어 벨트 완성판) 🏭

```python
chain = (
    {"context": retriever | RunnableLambda(split_image_text_types),   # 찾아서 그림/글로 나누고
     "question": RunnablePassthrough()}                               # 질문은 그대로 통과
    | RunnableLambda(img_prompt_func)                                 # 그림+글+질문으로 편지 조립
    | model                                                           # 눈 달린 AI에게 보내고
    | StrOutputParser()                                               # 답장에서 글자만 꺼내기
)
answer = chain.invoke(query)
```

```
질문 ─┬─▶ [사서: 찾기] ─▶ [그림/글 나누기] ─▶ context ─┐
      └─▶ [그대로 통과] ────────────────────▶ question ─┴─▶ [편지 조립] ─▶ [AI] ─▶ [글자만] ─▶ 답!
```

- 맨 앞 `{ ... }` 딕셔너리 → 질문 하나를 **두 갈래 길로 동시에** 보내요.
- `RunnablePassthrough()` → "아무것도 안 하고 **그대로 통과**". 질문 원본을 그대로 들고 가요.
- `RunnableLambda(함수)` → 내가 만든 보통 함수를 **컨베이어 벨트에 올릴 수 있게** 해주는 받침대.

### 핵심 코드 ⑥ — 편지 조립 함수 (img_prompt_func)

```python
def img_prompt_func(data_dict):
    messages = []
    for img in data_dict["context"]["images"]:          # ① 그림을 먼저 다 넣고
        messages.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    texts = "\n".join(data_dict["context"]["texts"])     # ② (반복문 밖!) 글을 합치고
    messages.append({"type": "text",
                     "text": "아래 자료를 근거로 질문에 한국어로 답하세요. "
                             "자료에 없는 내용은 모른다고 답하세요.\n"
                             f"질문: {data_dict['question']}\n참고 자료:\n{texts}"})
    return [HumanMessage(content=messages)]              # ③ 한 통의 편지로 돌려주기
```

- `"자료에 없는 내용은 모른다고 답하세요"` → AI가 **지어내지 않게(환각 방지)** 하는 약속.
- `"\n".join(목록)` → 목록 사이사이에 **줄바꿈**을 끼워 하나의 글로 합치기.

### Before / After 비교 (과제 7점!) 📊
같은 질문을 **글만 아는 RAG(Before)** 와 **그림도 보는 RAG(After)** 에 던져서 차이를 보여주는 것.
→ **그래프에만 있는 내용**을 물어봐야 차이가 확 보여요!

---

## 📕 코드 해석 사전 — 이것만 알면 코드가 읽혀요

| 코드 모양 | 읽는 법 | 예시 |
| --- | --- | --- |
| `def 이름(재료):` | 기계(함수) 만들기 | `def encode_image(image_path):` |
| `return 값` | 결과물 돌려주기 | `return texts, tables` |
| `[]` / `.append(x)` | 빈 상자 / 상자에 넣기 | `tables.append(...)` |
| `{"키": 값}` | 이름표 붙은 사물함(딕셔너리) | `{"images": ..., "texts": ...}` |
| `d["키"]` | 사물함에서 꺼내기 | `data["texts"]` |
| `for a in b:` | b에서 하나씩 꺼내 반복 | `for doc in docs:` |
| `if / elif / else` | 만약 / 아니면 만약 / 그 외 | |
| `[x for x in b]` | 반복해서 만든 걸 상자에 담기(한 줄) | `[str(uuid.uuid4()) for _ in ...]` |
| `lambda x: x` | 이름 없는 한 줄 기계 | |
| `f"...{변수}..."` | 빈칸에 변수 값 끼우기 | `f"{len(texts)}개"` |
| `with open(경로, "rb") as f:` | 파일 열고, 끝나면 자동 닫기 | |
| `A \| B \| C` | 컨베이어 벨트(A 결과 → B 재료) | `prompt \| model` |
| `.invoke(x)` / `.batch([...])` | 한 번 실행 / 여러 개 실행 | |
| `try: ... except 에러:` | 해 보고, 안 되면 다른 방법 | import 경로 바꾸기 |
| `raise ValueError(...)` | 문제 발생! 멈춰! | API 키 없을 때 |
| `Path(...) / "폴더"` | 경로 이어 붙이기 | `PROJECT_ROOT / "data"` |

### 코드 읽는 3단계 비법 🪄
1. **재료(입력)** 는 뭐지? → 함수 괄호 안, 또는 `|` 왼쪽
2. **무슨 일**을 하지? → 함수 이름, 주석(`#`)
3. **결과(출력)** 는 뭐지? → `return`, 또는 `|` 오른쪽으로 넘어가는 것

---

## 🖍️ 선생님의 빨간펜 — 03 노트북에서 고쳐야 할 곳

여러분 `03_임베딩_최종체인_연습.ipynb` 를 채점해 봤어요. 위의 정리 코드는 **고친 버전**이에요.

1. **`img_prompt_func` 들여쓰기 (가장 중요!)**
   `texts = ...`, `messages.append(글)`, `return` 이 `for` 반복문 **안**에 들어가 있어요.
   → 그림이 0장이면 아무것도 돌려주지 않고(`None`) **에러**, 그림이 여러 장이면 **첫 장만** 보고 끝나요. 세 줄을 왼쪽으로 한 칸 빼세요.
2. **f-문자열이 반만 f** — `f"아래 자료를..." "...질문: {data_dict['question']}..."`
   두 번째 따옴표 덩어리에 `f`가 없어서 `{질문}`, `{texts}` 가 **값이 아니라 글자 그대로** 들어가요. AI가 질문을 못 봐요! 두 번째 덩어리 앞에도 `f`를 붙이세요.
3. **`split_image_text_types` 의 첫 줄** — `doc.page_content if ... else doc` 를 계산만 하고 **어디에도 저장 안 했어요.** `doc = ...` 로 바꾸세요. (지금은 원본이 문자열이라 우연히 동작 중)
4. **`load_dotenv(".env")`** → 현재 폴더(practice)에서 찾아요. 위에서 찾은 `PROJECT_ROOT / ".env"` 를 쓰세요.
5. **import 실패 시 `print`만 함** — `except ImportError:` 안에서 `from langchain.retrievers.multi_vector import MultiVectorRetriever` 처럼 **진짜로 다른 경로를 import** 해야 해요.
6. **8번 Before/After, 9번 번역 체인 셀이 아직 비어 있어요.** 8번은 과제 7점이니 꼭!

---

## 💭 생각해 볼 질문 — 선생님 힌트

**[02] Q1. 그림을 왜 글로 요약할까?**
→ 검색은 **글끼리 뜻을 비교**하는 방식이라, 그림을 글(메모)로 바꿔야 질문과 비교할 수 있어요.

**[02] Q2. 요약만 저장하고 원본을 버리면?**
→ 메모는 짧아서 **숫자·세부 내용이 빠져요.** AI가 정확히 답하려면 진짜 원본(특히 그림 자체)이 필요해요.

**[02] Q3. `max_concurrency=100` 이면?**
→ 한꺼번에 너무 많이 보내서 AI 회사가 막아요(`RateLimitError`). 돈도 한 번에 많이 나가요.

**[03] Q1. 요약과 원본에 같은 `doc_id`를 안 붙이면?**
→ 메모는 찾았는데 **어느 책인지 몰라서 원본을 못 꺼내요.** 번호표 없는 도서관!

**[03] Q2. 검색은 요약으로, 답변은 원본으로 하는 이유?**
→ 요약은 **찾기 쉽고**, 원본은 **정보가 많아요.** 각자 잘하는 일을 맡긴 거예요. (색인 카드로 찾고, 책을 읽고 답하기)

**[03] Q3. 실제 서비스로 만들려면?**
→ 꺼지면 잊어버리는 `InMemoryStore` 대신 **파일·DB·Redis** 같은 **오래 기억하는 저장소**를 쓰고, Chroma도 `persist_directory` 로 디스크에 저장해요.

---

## 📒 단어장

| 단어 | 쉬운 뜻 |
| --- | --- |
| PDF 파싱/추출 | 책을 글·표·그림으로 오려 나누기 |
| 청크(chunk) | 잘라낸 한 조각 |
| 오버랩(overlap) | 조각끼리 겹치는 부분 |
| 토큰(token) | AI가 글을 세는 단위(낱말 조각) — 돈 계산의 기준 |
| LLM | 글을 읽고 쓰는 AI |
| VLM | 그림도 볼 수 있는 AI (눈 달린 AI) |
| 프롬프트 | AI에게 쓰는 부탁 편지 |
| Base64 | 그림을 글자로 포장한 것 |
| 임베딩 | 글의 뜻을 숫자 좌표로 바꾸기 |
| 벡터 DB (Chroma) | 숫자 좌표를 보관하고 가까운 걸 찾아주는 서랍 |
| docstore | 원본을 보관하는 책장 |
| doc_id | 요약과 원본을 잇는 번호표 |
| retriever | 찾아서 꺼내주는 사서 |
| 체인(LCEL, `\|`) | 일을 차례로 넘기는 컨베이어 벨트 |
| 환각 | AI가 모르는 걸 지어내는 것 |

> 🌟 **오늘의 한 줄 요약**
> **"오리고(추출) → 메모하고(요약) → 같은 번호표로 정리하고(임베딩) → 메모로 찾아 원본을 보고 대답한다(체인)!"**
