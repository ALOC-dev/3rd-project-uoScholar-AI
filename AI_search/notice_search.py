import mysql.connector
import json
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI

# 1. 환경 변수 및 OpenAI 클라이언트 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2. DB 연결 함수
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

# 3. 임베딩 함수
def embed_text(text: str) -> list:
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 실패: {text}")
        print("에러:\n", e)
        return []

# 4. 사용자 입력 포맷 변환 함수 (GPT 사용)
def format_user_query(user_input: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "사용자 질문을 다음 형식으로 바꿔줘: {학과명}에서 작성한 공지: {주제} ,가능한 한 간결하고 명확하게 작성해."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT 변환 실패, 원본 사용:", e)
        return f"학과에서 작성한 공지: {user_input}"
    
# 5. 코사인 유사도 계산
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 6. 유사 공지 검색
def search_similar_notices(user_input, top_n=5):
    formatted_input = format_user_query(user_input)
    query_vector = embed_text(formatted_input)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, department, link, posted_date, vector FROM notice WHERE vector IS NOT NULL")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    scored = []
    for row in rows:
        try:
            notice_vector = json.loads(row["vector"])
            score = cosine_similarity(query_vector, notice_vector)
            if score > 0.65:
                row["similarity"] = score
                scored.append(row)
        except Exception as e:
            print(f" 벡터 파싱 오류 (id={row['id']}):", e)

        scored.sort(key=lambda x: x["similarity"], reverse=True)

        cleaned = [
            {
                "title": row["title"],
                "department": row["department"],
                "link": row["link"],
                "posted_date": row["posted_date"]
            }
            for row in scored[:top_n]
        ]
        return cleaned

# 7. 실행부
if __name__ == "__main__":
    user_question = input("질문을 입력하세요: ")
    results = search_similar_notices(user_question)

    if not results:
        print(" 관련된 공지가 없습니다.")
    else:
        print("\n 유사 공지 결과:")
        for r in results:
            print(f"- {r['posted_date']} | {r['department']} | {r['title']}")
            print(f"  링크: {r['link']} | 유사도: {r['similarity']:.4f}")