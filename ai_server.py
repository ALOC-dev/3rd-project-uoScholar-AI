from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import mysql.connector
import os
import json
import re

# 1. 환경 변수 및 GPT 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = 
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 2. FastAPI 앱 생성
app = FastAPI()

# 3. 요청 바디 모델
class Question(BaseModel):
    question: str

# 4. MySQL 연결 함수
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

# 5. GPT 프롬프트로 키워드 추출
def extract_keywords(user_input: str) -> dict:
    prompt = f"""
너는 사용자의 자연어 질문을 이해하고, 다음 세 가지 항목으로 나누어 키워드를 JSON 형식으로 정리해야 해.

1. "title_keywords": 공지 제목에 들어갈 만한 **구체적이고 핵심적인 단어**만 (예: "장학금", "휴학", "등록금")  
   - "공지", "안내", "신청" 같은 **범용 단어는 제외해**  
   - **1~2개만** 추출해
2. "writer_keywords": 공지를 작성했을 가능성이 있는 부서, 혹은 학과 
3. "year": 특정 학년도나 연도가 명시된 경우 (숫자), 없으면 null

[사용자 질문]
"{user_input}"

[출력 형식 예시]
{{
  "title_keywords": ["장학금"],
  "writer_keywords": ["학생처"],
  "year": 2024
}}

설명 없이 JSON만 출력해줘. ```json 같은 마크다운도 제거해줘.
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    content = re.sub(r"```json|```", "", content).strip()

    try:
        return json.loads(content)
    except Exception as e:
        print("❌ GPT 응답 파싱 실패:", e)
        print("GPT 응답 내용:\n", content)
        return {"title_keywords": [], "writer_keywords": [], "year": None}

# DB에서 공지 검색 (여러 키워드 OR 조건 + 부서 LIKE)
def search_notices(title_keywords, year, department_keyword):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        title_conditions = " OR ".join(["title LIKE %s" for _ in title_keywords])
        query = f"""
        SELECT * FROM notice
        WHERE ({title_conditions})
          AND YEAR(posted_date) = %s
          AND department LIKE %s
        LIMIT 10;
        """
        params = [f"%{kw}%" for kw in title_keywords] + [year, f"%{department_keyword}%"]
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows

    except Exception as e:
        print("❌ DB 쿼리 에러:", e)
        return []
    
# 7. GPT가 DB 결과로 자연어 답변 생성
def generate_answer(question: str, search_results: list) -> str:
    if not search_results:
        return "요청하신 조건에 해당하는 공지를 찾을 수 없습니다."

    summarized = "\n\n".join(
        f"- {row['posted_date']} | {row['department']} | {row['title']}\n  📎 링크: {row['link']}"
        for row in search_results
    )

    prompt = f"""
[사용자 질문]
{question}

[관련 공지 목록]
{summarized}

위 공지들을 바탕으로 사용자의 질문에 자연스럽게 요약된 답변을 해줘.
너는 친절하고 정확한 챗봇이야. 공지 제목과 날짜, 부서를 적절히 참고해서 키워드가 들어가있는 항목들을 찾아서 요약해줘.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# 8. 메인 API 엔드포인트
@app.post("/analyze_question")
def analyze_question(q: Question):
    # 키워드 추출
    keywords = extract_keywords(q.question)

    # DB 검색
    title_keywords = keywords["title_keywords"]
    dept_keyword = keywords["writer_keywords"][0] if keywords["writer_keywords"] else ""
    year = keywords["year"] or 2024
    results = search_notices(title_keywords, year, dept_keyword)

    # GPT 답변 생성
    gpt_answer = generate_answer(q.question, results)

    return {
        "extracted_keywords": keywords,
        "search_results": results,
        "gpt_answer": gpt_answer
    }
