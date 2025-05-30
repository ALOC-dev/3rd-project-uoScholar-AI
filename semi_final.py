import mysql.connector
import json
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI

#  환경 변수 및 OpenAI 클라이언트 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"  # 필요시 수정
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#  DB 연결 함수
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

#  임베딩 함수 (OpenAI v1 방식)
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

#  코사인 유사도 계산
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#  유사 공지 검색
def search_similar_notices(user_input, top_n=5):
    query_vector = embed_text(user_input)

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
            row["similarity"] = score
            scored.append(row)
        except Exception as e:
            print(f" 벡터 파싱 오류 (id={row['id']}):", e)

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]

#  실행
if __name__ == "__main__":
    user_question = input("질문을 입력하세요: ")
    results = search_similar_notices(user_question)

    print("\n유사 공지 결과:")
    for r in results:
        print(f"- {r['posted_date']} | {r['department']} | {r['title']}")
        print(f"  링크: {r['link']} | 유사도: {r['similarity']:.4f}")
