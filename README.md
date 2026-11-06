git clone <저장소 주소>
cd multimodal-rag

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt

# 커널 등록
python -m ipykernel install --user --name multimodal-rag

# .env 작성
cp .env.example .env                # 윈도우: copy .env.example .env
# .env 파일을 열어 실제 API 키를 입력

## 실습 결과
======================================================================
질문: 24년말 대비 26년 1분기의 누적 평가이익을 알려줘
======================================================================
[BEFORE] 텍스트 전용 RAG
 자료에 따르면 2026년 1분기의 누적 평가이익에 대한 구체적인 수치는 제공되지 않았습니다. 따라서 24년말 대비 26년 1분기의 누적 평가이익에 대한 정보는 알 수 없습니다. 

[AFTER] 멀티모달 RAG
 26년 1분기의 누적 평가이익은 571조 원입니다. 