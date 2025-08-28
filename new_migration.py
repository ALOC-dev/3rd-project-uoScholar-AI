# index_from_mysql.py
import os, re, hashlib, mysql.connector
from dotenv import load_dotenv; load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ========= ENV =========
DB = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "connection_timeout": int(os.getenv("DB_CONN_TIMEOUT", "10")),
    "use_pure": os.getenv("DB_USE_PURE", "true").lower() == "true",
    "raise_on_warnings": os.getenv("DB_WARNINGS", "true").lower() == "true",
}

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "uos-notices")
PINECONE_NS      = os.getenv("PINECONE_NAMESPACE")  # 선택(없으면 기본)

BATCH_SIZE       = int(os.getenv("UPSERT_BATCH_SIZE", "200"))  # 100~300 권장
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))

# ========= Helpers =========
def norm_text(s: str) -> str:
    """간단 전처리: HTML태그 제거(대강), 공백 정리"""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)          # 아주 단순 태그 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_notices():
    sql = """
      SELECT id, category, department, link, post_number,
             posted_date, title, summary
      FROM notice
      ORDER BY posted_date DESC, id DESC
    """
    print("[DB] connect params:", {k: DB[k] for k in ["host","user","database","port","use_pure"]})
    conn = mysql.connector.connect(**DB)
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"[DB] fetched rows: {len(rows)}")
    return rows

def hash_id(*parts) -> str:
    return hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]

def build_docs(row, splitter):
    docs = []
    doc_id = row["id"]
    title  = norm_text(row.get("title") or "")
    text   = norm_text(row.get("summary") or "")

    meta_base = {
        "doc_id": doc_id,
        "id": doc_id,
        "title": title,
        "link": row.get("link"),
        "posted_date": str(row.get("posted_date")),
        "category": row.get("category"),
        "department": row.get("department"),
    }

    # (A) 타이틀 전용 문서 (짧은 질의 대응력)
    if title:
        docs.append(Document(
            page_content=title,
            metadata={**meta_base, "type": "title", "chunk_id": "title"}
        ))

    # (B) 본문 summary → 청킹 (평균 500자면 대부분 1청크)
    if text:
        for i, chunk in enumerate(splitter.split_text(text)):
            if not chunk.strip():
                continue
            docs.append(Document(
                page_content=chunk.strip(),
                metadata={**meta_base, "type": "summary", "chunk_id": f"ch{i}"}
            ))
    return docs

def main():
    assert OPENAI_API_KEY and PINECONE_API_KEY, "OPENAI/PINECONE 키 필요"

    # 1) Pinecone & Embedding
    _ = Pinecone(api_key=PINECONE_API_KEY)  # 인덱스는 미리 생성(metric=cosine, dim=1536)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX, embedding=embeddings, namespace=PINECONE_NS
    )
    print(f"[PC] index={PINECONE_INDEX}, ns={PINECONE_NS}")

    # 2) Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    # 3) DB → 문서화 → 업서트
    rows = fetch_notices()
    batch_docs, batch_ids = [], []
    total_docs = 0

    for row in rows:
        docs = build_docs(row, splitter)
        if not docs:
            continue
        for d in docs:
            uid = hash_id(d.metadata["doc_id"], d.metadata["type"], d.metadata["chunk_id"])
            batch_docs.append(d)
            batch_ids.append(uid)

        if len(batch_docs) >= BATCH_SIZE:
            vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
            total_docs += len(batch_docs)
            print(f"[UPSERT] +{len(batch_docs)} docs (total {total_docs})")
            batch_docs, batch_ids = [], []

    if batch_docs:
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        total_docs += len(batch_docs)
        print(f"[UPSERT] +{len(batch_docs)} docs (total {total_docs})")

    print(f"✅ Upserted notices: {len(rows)} rows → {total_docs} docs (title + chunks)")

if __name__ == "__main__":
    main()
