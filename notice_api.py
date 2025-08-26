# notice_api.py

from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import os, json, logging

import mysql.connector
from mysql.connector import pooling
from mysql.connector import Error as MySQLError
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

from dotenv import load_dotenv

# LangChain / Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore  # pip install langchain-pinecone
# 만약 위 모듈이 없다면: from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, exceptions as pc_exceptions

# ==============================
# 0) 환경/로깅
# ==============================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "uos-notices")

COS_THRESHOLD = float(os.getenv("COS_THRESHOLD", "0.6"))
MIN_TURNS     = int(os.getenv("MIN_USER_TURNS_FOR_FINAL", "2"))

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),  # 배포 환경에서 반드시 설정
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "connection_timeout": int(os.getenv("DB_CONN_TIMEOUT", "5")),
}

def _mask(v: str, keep: int = 2) -> str:
    if not v:
        return ""
    return v[:keep] + "*" * max(0, len(v) - keep)

def _safe_db_cfg_for_log():
    safe = dict(DB_CONFIG)
    safe["password"] = _mask(safe.get("password") or "")
    return safe

# ==============================
# 1) Lazy 초기화 핸들
# ==============================
_pool: Optional[pooling.MySQLConnectionPool] = None
_pc: Optional[Pinecone] = None
_vs: Optional[PineconeVectorStore] = None
_embeddings: Optional[OpenAIEmbeddings] = None
_llm: Optional[ChatOpenAI] = None

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
        logging.info("✅ OpenAIEmbeddings ready: %s", EMBED_MODEL)
    return _embeddings

def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        logging.info("✅ ChatOpenAI ready: %s", CHAT_MODEL)
    return _llm

def get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        # 필수값 검증
        if not DB_CONFIG["host"]:
            raise RuntimeError("DB_HOST is not set")
        if not DB_CONFIG["user"]:
            raise RuntimeError("DB_USER is not set")
        if not DB_CONFIG["database"]:
            raise RuntimeError("DB_NAME is not set")

        logging.info("Creating MySQL pool with config: %s", json.dumps(_safe_db_cfg_for_log(), ensure_ascii=False))
        _pool = pooling.MySQLConnectionPool(
            pool_name="uos_pool",
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            **DB_CONFIG
        )
        logging.info("✅ MySQL pool created")
    return _pool

def get_conn():
    return get_pool().get_connection()

# get_vectorstore()
def get_vectorstore() -> PineconeVectorStore:
    global _pc, _vs
    if _vs is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")
        if _pc is None:
            _pc = Pinecone(api_key=PINECONE_API_KEY)
            logging.info("✅ Pinecone client initialized")
        _vs = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=get_embeddings(),
            text_key="title"   # ✅ 메타데이터의 'title'을 page_content로 사용
        )
        logging.info("✅ PineconeVectorStore ready: %s", PINECONE_INDEX)
    return _vs


# ==============================
# 2) 검색/프롬프트 유틸
# ==============================
def summarize_queries(queries: List[str]) -> str:
    """LLM으로 누적 히스토리를 검색 질의로 요약 (공지 제목에 들어갈만한 키워드 포함)"""
    if len(queries) == 1:
        return queries[0]
    llm = get_llm()
    sys = (
        "너는 키워드 포함 검색 쿼리 생성기야. 사용자의 누적 대화를 이해하고, "
        "공지 '제목'에 나올 법한 핵심 단어(학기/부서/카테고리/프로그램명 등)를 포함한 짧고 명확한 질의로 요약해."
    )
    prompt = "\n".join(queries)
    out = llm.invoke([("system", sys), ("user", prompt)])
    return out.content.strip()

SYSTEM_FOLLOWUP = (
    "당신은 친절하고 효율적인 대학 공지 검색 도우미 '공지봇'입니다. "
    "질문이 모호할 경우, 최소한의 핵심만 묻는 추가 질문을 단 한 문장으로 하세요."
)
def gen_followup_question(user_summary: str) -> str:
    llm = get_llm()
    out = llm.invoke([
        ("system", SYSTEM_FOLLOWUP),
        ("user", f"사용자 요구 요약: {user_summary}\n한 문장으로 공손하게 물어봐.")
    ])
    return out.content.strip()

SYSTEM_FINAL = (
    "당신은 대학 공지사항 안내 도우미 '공지봇'입니다. "
    "제공된 메타데이터와 summary만 근거로, 추측 없이 간결한 한 문단 답변을 작성하세요. 존댓말 유지."
)
def compose_final_answer(q_text: str, notice: Dict[str, Any]) -> str:
    llm = get_llm()
    user_prompt = (
        f"사용자 질문 요약: {q_text}\n\n"
        f"[공지 메타]\n- 제목: {notice.get('title')}\n- 게시일: {notice.get('posted_date')}\n"
        f"- 부서/학과: {notice.get('department')}\n- 카테고리: {notice.get('category')}\n- 링크: {notice.get('link')}\n\n"
        f"[공지 summary]\n{(notice.get('summary') or '')[:2000]}\n\n"
        "위 정보만 사용해 핵심만 담아 한 문단으로 답하세요."
    )
    out = llm.invoke([("system", SYSTEM_FINAL), ("user", user_prompt)])
    return out.content.strip()

def _doc_to_meta_tuple(doc, score: float) -> Tuple[Dict[str, Any], float]:
    """LangChain Document -> (metadata dict, score) 정규화"""
    md = doc.metadata or {}
    # Pinecone Vector ID는 LangChain Document에 기본적으로 없으므로,
    # 업서트 시 metadata에 'id'를 넣어두는 것을 강력 권장.
    # 없을 경우, link로 DB lookup 폴백.
    return {
        "id": md.get("id"),  # 업서트 때 넣어둔 경우만 존재
        "title": md.get("title"),
        "link": md.get("link"),
        "posted_date": md.get("posted_date"),
        "category": md.get("category"),
        "department": md.get("department"),
    }, float(score)

def fetch_summary_from_db(meta: Dict[str, Any]) -> str:
    """id 우선, 없으면 link로 폴백해서 summary 조회"""
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        if meta.get("id"):
            cur.execute("SELECT summary FROM notice WHERE id=%s", (meta["id"],))
        else:
            # id 메타가 없다면 link로 조회 (고유 링크 전제)
            cur.execute("SELECT summary FROM notice WHERE link=%s", (meta.get("link"),))
        row = cur.fetchone()
        return row["summary"] if row else ""
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

# ==============================
# 3) FastAPI
# ==============================
app = FastAPI(title="UoS Notice Chat API (LangChain Retriever)", version="2.0.0")

class ChatTurn(BaseModel):
    history: List[str]
    user_message: str

class ChatReply(BaseModel):
    found: bool
    assistant_message: str
    top_score: float
    selected: Optional[Dict[str, Any]] = None
    refs: Optional[List[Dict[str, Any]]] = None

@app.get("/debug/env")
def debug_env():
    return {
        "OPENAI_MODEL": CHAT_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "PINECONE_INDEX": PINECONE_INDEX,
        "DB_HOST": DB_CONFIG["host"],
        "DB_USER": DB_CONFIG["user"],
        "DB_NAME": DB_CONFIG["database"],
        "DB_PORT": DB_CONFIG["port"],
        "HAS_DB_PASSWORD": bool(DB_CONFIG["password"]),
        "HAS_OPENAI_KEY": bool(OPENAI_API_KEY),
        "HAS_PINECONE_KEY": bool(PINECONE_API_KEY),
    }

@app.get("/health/db")
def health_db():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        cur.close()
        conn.close()
        return {"ok": True}
    except Exception as e:
        logging.exception("DB health check failed")
        return {"ok": False, "error": str(e)}

@app.get("/health/pinecone")
def health_pinecone():
    try:
        _ = get_vectorstore()  # 초기화만 확인
        return {"ok": True}
    except Exception as e:
        logging.exception("Pinecone health check failed")
        return {"ok": False, "error": str(e)}

@app.post("/chat", response_model=ChatReply)
def chat(turn: ChatTurn):
    # 1) 누적 히스토리 → 요약 쿼리
    queries = (turn.history or []) + [turn.user_message]
    combined = summarize_queries(queries)

    # 2) LangChain VectorStore로 검색 (점수 필요 → with_score API 사용)
    vs = get_vectorstore()
    # PineconeVectorStore: similarity_search_with_score 지원
    docs_with_scores = vs.similarity_search_with_score(combined, k=5)

    if not docs_with_scores:
        followup = gen_followup_question(turn.user_message)
        return ChatReply(found=False, assistant_message=followup, top_score=0.0, refs=[])

    # 3) 임계치 필터
    metas_scores = []
    for doc, score in docs_with_scores:
        md, sc = _doc_to_meta_tuple(doc, score)
        metas_scores.append((md, sc))
    valid = [(md, sc) for (md, sc) in metas_scores if sc >= COS_THRESHOLD]
    has_min_turns = len(queries) >= MIN_TURNS

    # refs (상위 3개)
    refs = []
    for (md, sc) in metas_scores[:3]:
        refs.append({
            "id": md.get("id"),
            "title": md.get("title"),
            "link": md.get("link"),
            "score": sc
        })

    if valid and has_min_turns:
        # 가장 점수 높은 문서 선택
        best_md, top_score = max(valid, key=lambda t: t[1])
        # DB에서 summary 조회 (id 우선, 없으면 link 폴백)
        try:
            summary = fetch_summary_from_db(best_md)
        except MySQLError as e:
            logging.exception("MySQL query failed")
            return ChatReply(
                found=False,
                assistant_message=f"DB 조회 중 오류가 발생했어요: {str(e)}",
                top_score=top_score,
                refs=refs
            )

        selected = {
            "id": best_md.get("id"),
            "title": best_md.get("title"),
            "link": best_md.get("link"),
            "posted_date": best_md.get("posted_date"),
            "category": best_md.get("category"),
            "department": best_md.get("department"),
            "summary": summary,
        }
        final_answer = compose_final_answer(combined, selected)
        return ChatReply(found=True, assistant_message=final_answer, top_score=top_score, selected=selected, refs=refs)

    # 임계치 미만이거나 최소 턴 미충족 → 추가 질문
    top_score = metas_scores[0][1]
    followup = gen_followup_question(combined)
    return ChatReply(found=False, assistant_message=followup, top_score=top_score, refs=refs)
