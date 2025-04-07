from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import mysql.connector
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)

# MySQL 연결
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port = "3306"
    )

@app.route('/best-match', methods=['POST'])
def best_match():
    user_input = request.json.get('question')

    prompt = f"""
    너는 사용자의 자연어 질문을 이해하고, 다음 세 가지 항목으로 나누어 키워드를 JSON 형식으로 정리해야 해.

    1. "title_keywords": 공지 제목에 들어갈 만한 핵심 단어들
    2. "writer_keywords": 공지를 작성했을 가능성이 있는 부서명
    3. "year": 특정 학년도나 연도가 명시된 경우 (숫자), 없으면 null

    [사용자 질문]
    "{user_input}"

    [출력 형식 예시]
    {{
      "title_keywords": ["딥러닝", "수업"],
      "writer_keywords": ["인공지능학부"],
      "year": 2024
    }}

    형식에 맞춰 JSON만 출력해줘.
    """

    gpt_response = llm.invoke(prompt)
    keywords = json.loads(gpt_response.content)

    title_keywords = keywords.get("title_keywords", [])
    writer_keywords = keywords.get("writer_keywords", [])
    year = keywords.get("year")

    title_cond = " OR ".join([f"title LIKE '%{kw}%'" for kw in title_keywords]) if title_keywords else "1=1"
    writer_cond = " OR ".join([f"writer LIKE '%{kw}%'" for kw in writer_keywords]) if writer_keywords else "1=1"
    year_cond = f"YEAR(date) = {year}" if year else "1=1"

    query = f"""
    SELECT title, url
    FROM posts
    WHERE ({title_cond})
    AND ({writer_cond})
    AND {year_cond}
    ORDER BY date DESC
    LIMIT 1;
    """

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()

    if result:
        return jsonify({
            "title": result["title"],
            "url": result["url"]
        })
    else:
        return jsonify({
            "message": "관련 게시물을 찾을 수 없습니다."
        }), 404

if __name__ == '__main__':
    app.run(debug=True)
