import mysql.connector
import openai
import json
from dotenv import load_dotenv
import os
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

def embed_text(text: str) -> list:
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"임베딩 실패: {text}")
        print("에러:", e)
        return []

def vectorize_all_notices():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, department, title FROM notice WHERE vector IS NULL")
    rows = cursor.fetchall()

    for notice_id, department, title in rows:
        combined_text = f"{department}에서 작성한 공지: {title}"
        vector = embed_text(combined_text)
        if vector:
            vector_str = json.dumps(vector)
            cursor.execute(
                "UPDATE notice SET vector = %s WHERE id = %s",
                (vector_str, notice_id)
            )
            print(f"저장 완료: id={notice_id}")
            time.sleep(0.5)

    conn.commit()
    cursor.close()
    conn.close()
    print("전체 벡터화 완료")

if __name__ == "__main__":
    vectorize_all_notices()
