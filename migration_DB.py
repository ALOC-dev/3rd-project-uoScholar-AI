#migration_DB.py
from dotenv import load_dotenv
import os

import mysql.connector
from openai import OpenAI
from langchain.schema import Document
from pinecone import Pinecone

load_dotenv()

# 환경 변수에서 API 키 가져오기
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#OpenAI KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#DB Config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "autocommit": os.getenv("DB_AUTOCOMMIT", "False") == "True",
    "use_pure": os.getenv("DB_USE_PURE", "True") == "True",
    "connection_timeout": int(os.getenv("DB_CONN_TIMEOUT", "10")),
    "raise_on_warnings": os.getenv("DB_WARNINGS", "True") == "True",
}

pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 이름 설정
index_name = "uos-notices"

# 인덱스가 없으면 새로 생성
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,   # OpenAI text-embedding-3-small 기준
        metric="cosine",  # 유사도 계산 방식 (cosine 추천)
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)


# MySQL 연결
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor(dictionary=True)

# 공지사항 가져오기
cursor.execute("""
    SELECT id, category, title, summary, link, posted_date, department
    FROM notice
""")
rows = cursor.fetchall()

# OpenAI 임베딩 함수
def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # 1536차원
    )
    return response.data[0].embedding

# Pinecone에 업서트
vectors = []
for row in rows:
    embedding = get_embedding(row["summary"])
    vectors.append({
        "id": str(row["id"]),
        "values": embedding,
        "metadata": {
            "title": row["title"],
            "link": row["link"],
            "category": row["category"],
            "posted_date": str(row["posted_date"]),
            "department": row["department"]
        }
    })

# 배치 업서트
index.upsert(vectors)
print("✅ MySQL 공지사항 → Pinecone 저장 완료")
