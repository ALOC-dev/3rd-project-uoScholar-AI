# 3rd-project-uoScholar-AI (Python)

## 프로젝트 소개
서울시립대 재학생들은 복수전공, 선후수 체계, 수강 제한, 학점 이수 기준 등 학사 관련 정보를 주로 공지사항에서 확인해야 하지만, 공지의 **가독성과 접근성이 떨어져** 원하는 정보를 빠르게 찾기 어렵습니다.  
이를 해결하기 위해 저희 팀은 필요한 공지를 쉽고 정확하게 제공하는 **챗봇 어플리케이션 "UoScholar"**를 개발하였습니다.

UoScholar는 서울시립대학교 공지사항 데이터를 크롤링하고, 임베딩을 통해 벡터로 DB에 저장합니다. 이후 LLM 기반 프롬프트 엔지니어링을 통해 학생들의 질문에 맞는 공지사항을 찾아내고, 이를 포함한 자연어 답변을 제공합니다.

---

## 전체 흐름도
1. 사용자가(Client) 자연어 형태로 질문을 입력한다.   
2. DB에서 질문과 가장 유사한 공지를 검색한다.  
3. 검색된 공지를 사용자에게 제공하는 동시에, 질문 의도에 맞는 자연스러운 응답으로 재구성한다.  
4. 완성된 답변을 Client로 전송하여 출력한다.  

---

## 주요 기능
- 📌 **공지사항 크롤링**: Playwright를 활용하여 서울시립대 공지사항을 PDF로 변환 후 LLM을 통해 내용 요약  
- 📌 **텍스트 임베딩 및 벡터화**: 해당 내용을 포함한 공지사항에 대한 정보를 벡터로 변환하여 DB에 저장  
- 📌 **LLM 응답 생성**: 사용자의 질문에 대해 DB에서 적합한 공지를 검색, 이를 포함한 자연스러운 답변을 LLM을 통해 제공  
- 📌 **챗봇 인터페이스**: 학생들이 채팅을 통해 학교 정보에 대한 질의응답 가능  

---
## 📌 설치 및 실행 방법

# 1. clone repository
git clone https://github.com/ALOC-dev/3rd-project-uoScholar-AI.git
cd 3rd-project-uoScholar-AI

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 3. requirements 설치
pip install -r requirements.txt

# 4. requirements.txt (예시)
fastapi
uvicorn
requests
beautifulsoup4
mysql-connector-python
openai
langchain-openai
python-dotenv
playwright

# 5. Playwright 브라우저 설치
playwright install chromium

# 6. .env 파일 설정
OPENAI_API_KEY=sk-xxxxxx

# 7. 크롤러 실행 (공지사항 크롤링 및 DB 저장)
python pdf_crawler.py

# 8. 서버 실행 (FastAPI)
uvicorn server:app --reload --host 0.0.0.0 --port 8000

---

## 📌 API Example

# Request
POST /analyze_question
Content-Type: application/json

{
  "question": "2024년 장학금 신청 공지 알려줘"
}

# Response
{
  "extracted_keywords": {
    "title_keywords": ["장학금"],
    "writer_keywords": ["학생처"],
    "year": 2024
  },
  "search_results": [
    {
      "posted_date": "2024-03-01",
      "department": "학생처",
      "title": "2024학년도 1학기 장학금 신청 안내",
      "link": "https://www.uos.ac.kr/..."
    }
  ],
  "gpt_answer": "2024학년도 1학기 장학금 신청은 학생처에서 진행되며 3월 1일부터 접수 시작됩니다."
}



---


## 기술 스택
- **Language**: Python  
- **Database**: MySQL 
- **LLM**: OpenAI GPT 기반  
- **크롤링**: Requests, BeautifulSoup, Playwright 
