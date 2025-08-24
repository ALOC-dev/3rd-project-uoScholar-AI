# run_retriever.py
from typing import List, Callable, Tuple, Dict
import os, json
import numpy as np
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from mysql.connector.pooling import MySQLConnectionPool
from pydantic import PrivateAttr
from pydantic import ConfigDict
import mysql.connector

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# =========================
# 0) DB 접속 정보 (그대로 export)
# =========================
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

# =========================
# 1) 임베딩 함수 (그대로 export)
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 .env 또는 환경변수에 없습니다.")
client = OpenAI(api_key=api_key)

def embed_fn(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vec = resp.data[0].embedding
    return np.array(vec, dtype=np.float32)

# =========================
# 2) 유틸
# =========================
def cos_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-8)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return Mn @ qn

# =========================
# 3) 리트리버 클래스 (그대로 export)
# =========================
class MySQLHybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn_uoscholar: Dict
    embed_fn: Callable[[str], np.ndarray]
    k: int = 5
    candidate_n: int = 200
    alpha: float = 0.7
    use_bm25_score: bool = True
    pool_name: str = "retriever_pool"
    pool_size: int = 5

    _pool: MySQLConnectionPool = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._pool = MySQLConnectionPool(
            pool_name=self.pool_name,
            pool_size=self.pool_size,
            **self.conn_uoscholar
        )

    def _fetch_candidates(self, query: str) -> List[dict]:
        sql = """
        SELECT
            id, category, post_number, title, summary, link, department,
            posted_date, embedding_vector,
            MATCH(title, summary) AGAINST (%s IN NATURAL LANGUAGE MODE) AS bm25
        FROM notice
        WHERE MATCH(title, summary) AGAINST (%s IN NATURAL LANGUAGE MODE)
        ORDER BY posted_date DESC, id DESC
        LIMIT %s
        """
        cnx = self._pool.get_connection()
        try:
            cur = cnx.cursor(dictionary=True)
            cur.execute(sql, (query, query, self.candidate_n))
            rows = cur.fetchall()
            cur.close()
            return rows
        finally:
            cnx.close()

    def _rerank(self, query_vec: np.ndarray, rows: List[dict]):
        vecs = []
        for r in rows:
            try:
                v = np.array(json.loads(r["embedding_vector"]), dtype=np.float32)
            except Exception:
                v = None
            vecs.append(v)

        valid_idx = [i for i, v in enumerate(vecs) if isinstance(v, np.ndarray)]
        if not valid_idx:
            order = np.arange(len(rows))[:self.k]
            return [(int(i), 0.0) for i in order]

        M = np.stack([vecs[i] for i in valid_idx], axis=0)
        cos_scores = cos_sim_matrix(query_vec.astype(np.float32), M)

        final_scores = np.full(len(rows), -1e9, dtype=np.float32)
        for j, i in enumerate(valid_idx):
            final_scores[i] = float(cos_scores[j])

        top_idx = np.argsort(-final_scores)[:self.k]
        return [(int(i), float(final_scores[i])) for i in top_idx]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        q_vec = self.embed_fn(query).astype(np.float32)
        rows = self._fetch_candidates(query)
        if not rows:
            return []
        ranked = self._rerank(q_vec, rows)
        docs: List[Document] = []
        for idx, score in ranked:
            r = rows[idx]
            docs.append(
                Document(
                    page_content=r.get("summary") or "",
                    metadata={
                        "id": r["id"],
                        "title": r["title"],
                        "link": r.get("link"),
                        "category": r.get("category"),
                        "post_number": r.get("post_number"),
                        "department": r.get("department"),
                        "posted_date": str(r.get("posted_date") or ""),
                        "score": score,
                    },
                )
            )
        return docs

    def debug_dump_first_stage(self, query: str) -> List[dict]:
        rows = self._fetch_candidates(query)
        print("\n=== [DEBUG] 1차 후보 (FULLTEXT 결과) ===")
        print(f"query='{query}' | candidate_n(limit)={self.candidate_n} | fetched={len(rows)}")
        print("정렬 기준: posted_date DESC, id DESC (동일 날짜면 id가 클수록 최신)")
        if not rows:
            print("(후보 없음)")
            return rows
        for rank, r in enumerate(rows, start=1):
            has_emb = bool(r.get("embedding_vector"))
            bm25 = r.get("bm25")
            bm25_s = f"{bm25:.6f}" if bm25 is not None else "N/A"
            print(f"[{rank}] id={r['id']:>6} | date={r.get('posted_date','')} | bm25={bm25_s} | "
                  f"emb={'Y' if has_emb else 'N'} | cat={r.get('category')} | post_no={r.get('post_number')}")
            print(f"     title: {r.get('title','')}")
            print(f"     link : {r.get('link','')}")
        return rows

# =========================
# 4) 외부에서 쓰기 쉬운 팩토리 (추가)
# =========================
def make_retriever(k: int = 5, candidate_n: int = 200) -> MySQLHybridRetriever:
    return MySQLHybridRetriever(
        conn_uoscholar=DB_CONFIG,
        embed_fn=embed_fn,
        k=k,
        candidate_n=candidate_n,
    )

# =========================
# 5) 데모 실행(직접 실행할 때만 동작)
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="검색 쿼리")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    # 연결 테스트
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ DB 연결 성공:", conn.is_connected())
        conn.close()
    except mysql.connector.Error as err:
        print("❌ DB 연결 실패:", err)
        raise

    retriever = make_retriever(k=args.k, candidate_n=args.n)
    retriever.debug_dump_first_stage(args.query)
    docs = retriever.invoke(args.query)
    print("\n=== [RERANK] Top-k Results ===")
    for i, d in enumerate(docs, start=1):
        md = d.metadata
        print(f"[{i}] score={md['score']:.4f} | {md.get('posted_date','')} | {md.get('title','')}")
        print(f"     link: {md.get('link')}")
