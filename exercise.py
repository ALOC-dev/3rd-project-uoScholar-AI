from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import mysql.connector
import json
import re

# MySQL 연결
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

def extract_keywords(user_input):
    prompt = f"""
    다음 사용자 질문에서 핵심 키워드를 추출해줘
    - 주제나 목적과 관련된 단어들을 중심으로
    - 단어 목록만 출력해줘. 설명 없이 JSON 배열로만 출력해줘
    - 대학교 학과 학부 이름이 적어도 1개는 포함되게 해줘

    **JSON 배열만 출력해줘. 설명, 줄바꿈, 예시 없이.**
    예시 출력: ["인공지능학과", "2학기", "복수전공", ...]

    질문: "{user_input}"
    """

    response = llm.invoke(prompt)

    return json.loads(response.content)

def search_notice_by_keywords(keywords):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    title_condition = " OR ".join([f"title LIKE '%{kw}%'" for kw in keywords])
    dept_condition = " OR ".join([f"department LIKE '%{kw}%'" for kw in keywords])
    where_clause = f"({title_condition}) AND ({dept_condition})"

    query = f"""
    SELECT title, department, link
    FROM notice
    WHERE {where_clause}
    LIMIT 20;
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return results

def safe_json_parse(content):
    try:
        # JSON 배열만 추출
        json_str = re.search(r'\[.*\]', content, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print("JSON 파싱 실패:", e)
        print("응답 내용:", content)
        return []

def ask_gpt_to_select(user_input, candidate_posts):
    post_list = "\n".join([
        f"- 제목: {p['title']}, 부서: {p['department']}, 링크: {p['link']}"
        for p in candidate_posts
    ])

    prompt = f"""
            다음은 사용자 질문과 관련 있을 수 있는 공지 목록입니다.

            사용자 질문: "{user_input}"

            공지 목록:
            {post_list}

            이 중에서 **사용자의 질문과 가장 관련 있는 공지 3개**를 골라줘.
            각 공지의 [부서, 제목, 링크]를 JSON 배열 형식으로만 보여줘.
            니 맘대로 url 만들지 말고 기존에 mySQL에 있는 링크를 보여줘줘
            **JSON 배열만 출력해줘. 설명, 줄바꿈, 예시 없이.**

            예시 출력: [{{"title": "공지 제목", "department": "학과", "link": "공지 링크"}},...]
            """


    response = llm.invoke(prompt)

    return safe_json_parse(response.content)

def print_results(results):
    if not results:
        print("관련 공지를 찾을 수 없습니다.")
        return
    
    print("\n 사용자 입력 관련 공지:")
    
    for row in results:
        print(f"\n제목: {row['title']}")
        print(f"부서: {row['department']}")
        print(f"링크: {row['link']}")

load_dotenv()
os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(model = "gpt-4o", temperature= 0.2)

user_input = input("질문을 입력해주세요 : ").strip()

keywords = extract_keywords(user_input)

candidate_posts = search_notice_by_keywords(keywords)

final_results = ask_gpt_to_select(user_input, candidate_posts)

print_results(final_results)