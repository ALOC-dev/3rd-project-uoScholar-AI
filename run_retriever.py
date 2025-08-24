# run_retriever.py (Query Rewrite + Cross-Encoder Reranker)

from typing import List, Callable, Tuple, Dict, Optional
import os, json, re
import numpy as np
from datetime import datetime
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
# 0) DB 접속 정보 (.env)
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
# 1) OpenAI 클라이언트
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 .env 또는 환경변수에 없습니다.")
oclient = OpenAI(api_key=api_key)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def embed_fn(text: str) -> np.ndarray:
    out = oclient.embeddings.create(model=EMBED_MODEL, input=text or "")
    return np.array(out.data[0].embedding, dtype=np.float32)

# =========================
# 2) Query Rewrite (LLM)
# =========================
def rewrite_query_for_search(query: str) -> str:
    """LLM으로 자연어 질문을 검색 친화적 문장으로 리라이트"""
    prompt = f"""
    사용자의 질문을 공지 검색에 최적화된 간결한 질의문으로 바꿔.
    - 의미는 유지하되 불필요한 말투 제거
    - 학과명, 제도명, 핵심 키워드는 반드시 보존
    - 짧고 간결한 문장으로

    예시:
    입력: "나 기계정보공학과로 전과하고 싶어."
    출력: "기계정보공학과 전과 안내"

    입력: "{query}"
    출력:
    """
    try:
        resp = oclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "당신은 검색 질의 최적화기입니다."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip().replace("\n", " ")
    except Exception as e:
        print("❌ Query rewrite 실패:", e)
        return query

# =========================
# 3) Cross-Encoder Reranker
# =========================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
_cross_encoder = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cross_encoder.to(_device)

def cross_encode_score(query: str, docs: List[str]) -> List[float]:
    pairs = [(query, d) for d in docs]
    inputs = _tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = _cross_encoder(**inputs).logits.squeeze(-1)
    return logits.cpu().tolist()

# =========================
# 4) 유틸
# =========================
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

def _recency_score(posted_date: Optional[str], tau_days: int = 30) -> float:
    if not posted_date:
        return 0.0
    try:
        d = datetime.fromisoformat(str(posted_date).replace(" ", "T"))
    except:
        try:
            d = datetime.strptime(str(posted_date), "%Y-%m-%d")
        except:
            return 0.0
    days = (datetime.now() - d).days
    return float(np.exp(-max(0, days) / float(tau_days)))

# =========================
# 5) 리트리버 클래스
# =========================
class MySQLHybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn_uoscholar: Dict
    embed_fn: Callable[[str], np.ndarray] = embed_fn

    k: int = 5
    candidate_n: int = 100
    recency_tau: int = 30

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

    # ---------- 1차 후보 (코사인 기반) ----------
    def _fetch_candidates_by_vector(self, query: str, limit: int = 5000) -> List[dict]:
        q_vec = self.embed_fn(query)

        sql = f"""
        SELECT id, category, post_number, title, summary, link, department,
               posted_date, embedding_vector
        FROM notice
        ORDER BY posted_date DESC
        LIMIT %s
        """
        cnx = self._pool.get_connection()
        try:
            cur = cnx.cursor(dictionary=True)
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
            cur.close()
        finally:
            cnx.close()

        scored = []
        for r in rows:
            v = None
            if r.get("embedding_vector"):
                try:
                    v = np.array(json.loads(r["embedding_vector"]), dtype=np.float32)
                except:
                    v = None
            if v is None:
                text = (r.get("summary") or "") + " " + (r.get("title") or "")
                v = self.embed_fn(text)

            # ✅ 차원 불일치 방어
            if v.shape[0] != q_vec.shape[0]:
                print(f"⚠️ 차원 불일치 무시됨: notice_id={r['id']} dim={v.shape[0]}")
                continue

            cos = cos_sim(q_vec, v)
            scored.append((cos, r))

        scored = sorted(scored, key=lambda x: -x[0])[: self.candidate_n]
        return [r for _, r in scored]

    # ---------- 2차 rerank ----------
    def _rerank_semantic(self, query: str, rows: List[dict]) -> List[Tuple[int, float]]:
        if not rows:
            return []

        texts = [(r.get("summary") or "").strip() or (r.get("title") or "") for r in rows]
        ce_scores = cross_encode_score(query, texts)

        scores = np.zeros(len(rows), dtype=np.float32)
        for i, r in enumerate(rows):
            rec = _recency_score(str(r.get("posted_date") or ""), tau_days=self.recency_tau)
            final = 0.9 * ce_scores[i] + 0.1 * rec
            scores[i] = float(final)

        top_idx = np.argsort(-scores)[: self.k]
        return [(int(i), float(scores[i])) for i in top_idx]

    # ---------- LangChain entry ----------
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # ✅ Query Rewrite
        rewritten_query = rewrite_query_for_search(query)
        print(f"🔍 Rewritten Query: {rewritten_query}")

        # 1차 후보
        rows = self._fetch_candidates_by_vector(rewritten_query, limit=5000)
        if not rows:
            return []

        # 2차 rerank
        ranked = self._rerank_semantic(rewritten_query, rows)

        docs: List[Document] = []
        for idx, score in ranked:
            r = rows[idx]
            docs.append(
                Document(
                    page_content=(r.get("summary") or r.get("title") or ""),
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

    # ---------- Debug ----------
    def debug_dump_first_stage(self, query: str) -> List[dict]:
        rewritten_query = rewrite_query_for_search(query)
        rows = self._fetch_candidates_by_vector(rewritten_query, limit=5000)

        print("\n=== [DEBUG] 1차 후보 (코사인 기반) ===")
        print(f"query='{query}' | rewritten='{rewritten_query}' | candidate_n={self.candidate_n} | fetched={len(rows)}")
        if not rows:
            print("(후보 없음)")
            return rows
        for rank, r in enumerate(rows[:20], start=1):
            print(f"[{rank}] id={r['id']:>6} | date={r.get('posted_date','')} "
                  f"| cat={r.get('category')} | post_no={r.get('post_number')}")
            print(f"     title: {r.get('title','')}")
        return rows

# =========================
# 팩토리
# =========================
def make_retriever(k: int = 5, candidate_n: int = 100, recency_tau: int = 30) -> MySQLHybridRetriever:
    return MySQLHybridRetriever(
        conn_uoscholar=DB_CONFIG,
        embed_fn=embed_fn,
        k=k,
        candidate_n=candidate_n,
        recency_tau=recency_tau,
    )

# =========================
# 실행 예시
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="검색 쿼리(자연어)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    retriever = make_retriever(k=args.k, candidate_n=args.n)
    retriever.debug_dump_first_stage(args.query)
    docs = retriever.invoke(args.query)

    print("\n=== [RERANK] Top-k Results ===")
    for i, d in enumerate(docs, start=1):
        md = d.metadata
        print(f"[{i}] score={md['score']:.4f} | {md.get('posted_date','')} | {md.get('title','')}")
