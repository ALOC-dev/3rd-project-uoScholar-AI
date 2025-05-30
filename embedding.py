import mysql.connector
from openai import OpenAI
import json
import os
import time
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = "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"

# 1. 환경 변수 불러오기
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. DB 연결 함수
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="dongha1005!",
        database="uoscholar_db",
        port=3306
    )

# 3. 텍스트 임베딩 함수 (최신 OpenAI API 방식)
def embed_text(text: str) -> list:
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ 임베딩 실패: {text}")
        print("에러:\n", e)
        return []

# 4. 전체 공지 벡터화 및 DB에 저장
def vectorize_all_notices():
    conn = get_connection()
    cursor = conn.cursor()

    # 아직 벡터화되지 않은 공지만 가져오기
    cursor.execute("SELECT id, department, title FROM notice")
    rows = cursor.fetchall()

    for notice_id, department, title in rows:
        combined_text = f"{department}에서 작성한 공지: {title}"
        embedding = embed_text(combined_text)

        if embedding:
            embedding_str = json.dumps(embedding)  # 리스트 → 문자열
            cursor.execute(
                "UPDATE notice SET vector = %s WHERE id = %s",
                (embedding_str, notice_id)
            )
            print(f"✅ 저장 완료: id={notice_id}")
            time.sleep(0.5)  # 속도 제한 방지

    conn.commit()
    cursor.close()
    conn.close()
    print("🎉 전체 임베딩 저장 완료!")

# 5. 실행
if __name__ == "__main__":
    vectorize_all_notices()
