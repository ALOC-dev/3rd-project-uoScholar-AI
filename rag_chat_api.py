# rag_chat_api.py
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from functools import lru_cache
from math import sqrt
import os, re, json, logging

from fastapi import FastAPI
from dotenv import load_dotenv; load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ===== Env / Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "uos-notices")
PINECONE_NS      = os.getenv("PINECONE_NAMESPACE")

# 검색 파라미터
USE_HYDE        = os.getenv("USE_HYDE", "true").lower() == "true"
COS_THRESHOLD   = float(os.getenv("COS_THRESHOLD", "0.60"))
TOP_K           = int(os.getenv("TOP_K", "12"))
FETCH_K         = int(os.getenv("FETCH_K", "120"))
LAMBDA_MMR      = float(os.getenv("LAMBDA_MMR", "0.4"))
TITLE_BOOST     = float(os.getenv("TITLE_BOOST", "0.35"))

# 자동 동의어 JSON 경로
AUTO_SYNS_PATH = os.getenv("AUTO_SYNS_PATH", "data/auto_synonyms.json")

# ===== Conversational Builder Params =====
CONV_INCLUDE_ASSISTANT = os.getenv("CONV_INCLUDE_ASSISTANT", "false").lower() == "true"  # 기본: user만
MAX_TURNS_FOR_CONTEXT  = int(os.getenv("MAX_TURNS_FOR_CONTEXT", "6"))
MAX_CONV_CHARS         = int(os.getenv("MAX_CONV_CHARS", "300"))
USE_HYDE_FOR_CONV      = os.getenv("USE_HYDE_FOR_CONV", "false").lower() == "true"       # 기본: 끔

# ===== Singletons =====
_pc = None
_vs = None
_emb = None
_llm = None
_auto_syns: Dict[str, List[str]] = {}

def get_embeddings():
    global _emb
    if _emb is None:
        assert OPENAI_API_KEY, "OPENAI_API_KEY 필요"
        _emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
        logging.info("✅ Embeddings ready: %s", EMBED_MODEL)
    return _emb

def get_llm():
    global _llm
    if _llm is None:
        assert OPENAI_API_KEY, "OPENAI_API_KEY 필요"
        _llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        logging.info("✅ Chat LLM ready: %s", CHAT_MODEL)
    return _llm

def get_vectorstore():
    global _pc, _vs
    if _vs is None:
        assert PINECONE_API_KEY, "PINECONE_API_KEY 필요"
        if _pc is None:
            _pc = Pinecone(api_key=PINECONE_API_KEY)
            logging.info("✅ Pinecone client initialized")
        _vs = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=get_embeddings(),
            namespace=PINECONE_NS
        )
        logging.info("✅ PineconeVectorStore ready: %s (ns=%s)", PINECONE_INDEX, PINECONE_NS)
    return _vs

def load_auto_syns() -> Dict[str, List[str]]:
    global _auto_syns
    if _auto_syns:
        return _auto_syns
    try:
        with open(AUTO_SYNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            _auto_syns = data.get("synonyms", {})
            logging.info("✅ Loaded auto synonyms: %s (terms=%d)", AUTO_SYNS_PATH, len(_auto_syns))
    except Exception as e:
        logging.warning("No auto synonyms loaded: %s", e)
        _auto_syns = {}
    return _auto_syns

# ===== Query Utils =====
# 불용어 확장(구어체 포함)
STOPWORDS = {
    "공지","안내","프로그램","워크숍","행사","공지사항","공지요","문의","신청",
    "관련","관련된","있어","있나요","혹시","좀","요","거","것","같아","싶어","겨","부터","까지"
}

def _basic_tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[0-9A-Za-z가-힣]+", (s or "").lower()) if t]

def _clean_query(q: str) -> str:
    toks = [t for t in _basic_tokens(q) if t not in STOPWORDS]
    return " ".join(toks) if toks else (q or "").strip()

def hyde_expand(q: str) -> str:
    sys = "대학 공지에 대한 가능한 요약 답변(1~2문장)을 한국어로 가정해서 써줘. 핵심 키워드를 포함해."
    hyp = get_llm().invoke([("system", sys), ("user", q)]).content.strip()
    return f"{q}\n\n[가상요약]\n{hyp}"

def expand_with_auto(tokens: List[str]) -> List[str]:
    syns = load_auto_syns()
    base = [(t or "").lower() for t in tokens if t]
    out, seen = [], set()
    for t in base:
        if t and t not in seen:
            out.append(t); seen.add(t)
        for s in syns.get(t, []):
            if s and s not in seen:
                out.append(s); seen.add(s)
    return out

# allow_hyde 토글 지원
def make_query(q: str, allow_hyde: bool = True) -> str:
    cleaned = _clean_query(q)
    toks = _basic_tokens(cleaned)
    toks_expanded = expand_with_auto([t for t in toks if t not in STOPWORDS])
    expanded_query = " ".join(toks_expanded) if toks_expanded else cleaned

    # 대화 합성 쿼리·장문·줄바꿈 포함 시 HyDE 금지
    if not allow_hyde or len(expanded_query) > 120 or "\n" in expanded_query:
        return expanded_query

    if len(toks_expanded) <= 2:
        return expanded_query  # 짧은 질의는 HyDE 비활성화
    if USE_HYDE:
        sys = "대학 공지에 대한 가능한 요약 답변(1~2문장)을 한국어로 가정해서 써줘. 핵심 키워드를 포함해."
        hyp = get_llm().invoke([("system", sys), ("user", expanded_query)]).content.strip()
        return f"{expanded_query}\n\n[가상요약]\n{hyp}"
    return expanded_query

def cosine_sim(a, b, eps: float = 1e-10) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)) + eps
    nb = sqrt(sum(y*y for y in b)) + eps
    return dot / (na * nb)

def _pc_filter(flt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not flt:
        return None
    out: Dict[str, Any] = {}
    for k, v in flt.items():
        if isinstance(v, dict):
            out[k] = v
        else:
            out[k] = {"$eq": v}
    return out

def _title_token_overlap_boost(query: str, title: str, boost: float) -> float:
    if not title or not query: return 0.0
    qt = set(_basic_tokens(query))
    tt = set(_basic_tokens(title))
    if not qt or not tt: return 0.0
    overlap = len(qt & tt) / max(1, len(qt))
    return overlap * boost

def _lexical_token_boost(query: str, title: str,
                         per_token_partial: float = 0.20,
                         per_token_exact: float = 0.60) -> float:
    """자동 확장 토큰 기반 부분/정확 매치 보너스."""
    base = [t for t in _basic_tokens(query) if t not in STOPWORDS]
    qtoks = expand_with_auto(base)
    if not qtoks or not title:
        return 0.0
    t = (title or "").lower()
    score = 0.0
    for qt in qtoks:
        if qt in t:
            score += per_token_partial
        if re.search(rf"(?:^|[^0-9A-Za-z가-힣]){re.escape(qt)}(?:$|[^0-9A-Za-z가-힣])", t):
            score += per_token_exact
    return min(score, 1.2)

# ===== Retrieval =====
def _mmr_docs(query: str, k: int, fetch_k: int, lambda_mult: float, flt: Optional[Dict]=None):
    vs = get_vectorstore()
    pcflt = _pc_filter(flt)
    logging.info("[MMR] query=%r k=%d fetch_k=%d lambda=%.2f filter=%s",
                 query, k, fetch_k, lambda_mult, pcflt)
    return vs.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=pcflt
    )

def _score_docs(query: str, docs):
    if not docs:
        return []
    emb = get_embeddings()
    qv = emb.embed_query(query)
    doc_vecs = emb.embed_documents([d.page_content for d in docs])
    return [(doc, float(cosine_sim(qv, dv))) for doc, dv in zip(docs, doc_vecs)]

@lru_cache(maxsize=256)
def _title_first(query: str, k: int, fetch_k: int):
    docs = _mmr_docs(query, k=k, fetch_k=fetch_k, lambda_mult=LAMBDA_MMR, flt={"type": "title"})
    scored = _score_docs(query, docs)
    boosted = []
    for d, sc in scored:
        md = d.metadata or {}
        title = md.get("title") or d.page_content or ""
        sc2 = sc
        sc2 += _title_token_overlap_boost(query, title, TITLE_BOOST)
        sc2 += _lexical_token_boost(query, title, per_token_partial=0.25, per_token_exact=0.35)
        boosted.append((d, sc2))
    boosted.sort(key=lambda t: t[1], reverse=True)
    return boosted

@lru_cache(maxsize=256)
def _summary_mmr(query: str, k: int, fetch_k: int):
    docs = _mmr_docs(query, k=k, fetch_k=fetch_k, lambda_mult=LAMBDA_MMR, flt={"type": "summary"})
    scored = _score_docs(query, docs)
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

def group_by_doc(docs_scores):
    by_doc: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"best": None, "chunks": []})
    for doc, sc in docs_scores:
        md = doc.metadata or {}
        doc_id = str(md.get("doc_id"))
        item = {"doc": doc, "score": float(sc)}
        by_doc[doc_id]["chunks"].append(item)
        if (by_doc[doc_id]["best"] is None) or (sc > by_doc[doc_id]["best"]["score"]):
            by_doc[doc_id]["best"] = item
    reps = []
    for doc_id, pack in by_doc.items():
        reps.append((doc_id, pack["best"]["score"], pack))
    reps.sort(key=lambda x: x[1], reverse=True)
    return reps

# ===== LLM Answer Style =====
SYSTEM_FINAL = (
    "당신은 대학 공지사항 안내 도우미 '공지봇'입니다. "
    "찾은 공지의 제목을 소개하고, 제공된 메타데이터와 summary만 근거로 summary에 대해 자세하게 설명해주세요." \
    "예시) 제가 찾아낸 공지는 ~~~~입니다. 이 공지는 ~~~~~입니다. 자세한 내용은 제공된 링크를 통해 확인할 수 있습니다."
)

def build_context(chunks_pack, max_chars=1800):
    chunks = sorted(chunks_pack["chunks"], key=lambda e: e["score"], reverse=True)[:3]
    ctx, size = [], 0
    for c in chunks:
        text = (c["doc"].page_content or "").strip()
        if not text: continue
        add_len = min(len(text), max_chars - size)
        if add_len <= 0: break
        ctx.append(text[:add_len]); size += add_len
        if size >= max_chars: break
    return "\n---\n".join(ctx)

def doc_meta(pack):
    md = pack["best"]["doc"].metadata or {}
    return {
        "id": md.get("id"),
        "doc_id": md.get("doc_id"),
        "title": md.get("title"),
        "link": md.get("link"),
        "posted_date": md.get("posted_date"),
        "category": md.get("category"),
        "department": md.get("department"),
    }

def enrich_pack_with_summaries(best_pack: Dict[str, Any], query: str, top_chunks: int = 12) -> Dict[str, Any]:
    """선택된 doc_id의 summary 청크를 추가로 수집/점수화해서 합친다."""
    try:
        doc_id = str(best_pack["best"]["doc"].metadata.get("doc_id"))
    except Exception:
        return best_pack

    vs = get_vectorstore()
    pairs = vs.similarity_search(
        query, k=top_chunks, filter=_pc_filter({"type": "summary", "doc_id": doc_id})
    )
    scored = _score_docs(query, pairs)
    merged = {"best": best_pack["best"], "chunks": list(best_pack["chunks"])}
    for d, sc in scored:
        merged["chunks"].append({"doc": d, "score": float(sc)})
    return merged

# ===== Conversational Query Builder (정확도 개선판) =====
def _gather_all_msgs(turn: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    messages / history / user_message 를 모두 모아서 시간순으로 반환.
    """
    msgs: List[Dict[str, str]] = []
    if isinstance(turn, dict):
        if isinstance(turn.get("messages"), list):
            msgs.extend([{"role": (m.get("role") or ""), "content": (m.get("content") or "")} for m in turn["messages"]])
        if isinstance(turn.get("history"), list):
            msgs.extend([{"role": (m.get("role") or ""), "content": (m.get("content") or "")} for m in turn["history"]])
        if turn.get("user_message"):
            msgs.append({"role": "user", "content": (turn["user_message"] or "")})
    return msgs

def _compose_conv_text(msgs: List[Dict[str, str]], max_turns: int) -> str:
    """
    최근 max_turns의 'user' 발화(또는 옵션에 따라 assistant 포함)를 role 태그 없이 공백으로 연결.
    뒤에서 자르는 하드 컷으로 최신 정보 유지.
    """
    msgs = [m for m in msgs if (m.get("content") or "").strip()]
    if not msgs:
        return ""
    if not CONV_INCLUDE_ASSISTANT:
        msgs = [m for m in msgs if (m.get("role") or "").lower() == "user"]
    recent = msgs[-max_turns:]
    text = " ".join([(m.get("content") or "").strip() for m in recent]).strip()
    if len(text) > MAX_CONV_CHARS:
        text = text[-MAX_CONV_CHARS:]
    return re.sub(r"\s+", " ", text)

def build_conversational_query(turn: Dict[str, Any]) -> Tuple[str, bool]:
    """
    대화 기반 질의와 'HyDE 허용 여부'를 반환.
    대화 합성 쿼리는 기본적으로 HyDE 비활성(환경변수로만 켤 수 있음).
    """
    msgs = _gather_all_msgs(turn)
    if not msgs:
        return "", False
    conv = _compose_conv_text(msgs, MAX_TURNS_FOR_CONTEXT)
    return conv, bool(USE_HYDE_FOR_CONV)

def _make_selected_with_summary(best_pack: Dict[str, Any], max_chars: int = 400) -> Dict[str, Any]:
    """
    selected에 summary 필드를 추가. summary 청크 상위 몇 개를 합쳐 짧게 제공.
    """
    m = doc_meta(best_pack)
    chunks = sorted(best_pack["chunks"], key=lambda e: e["score"], reverse=True)
    texts = []
    for c in chunks:
        md = (c["doc"].metadata or {})
        if md.get("type") == "summary":
            t = (c["doc"].page_content or "").strip()
            if t:
                texts.append(t)
        if len(texts) >= 3:
            break
    merged = " ".join(texts).strip()
    if len(merged) > max_chars:
        merged = merged[:max_chars].rstrip() + "…"
    return {**m, "summary": merged}

# ===== FastAPI =====
app = FastAPI(title="UoS RAG Chat API", version="1.9.0")

@app.get("/health")
def health():
    try:
        _ = get_vectorstore()
        _ = load_auto_syns()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/stats")
def debug_stats():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX)
    def count(flt):
        st = idx.describe_index_stats(filter=_pc_filter(flt), namespace=PINECONE_NS)
        return st.get("namespaces", {}).get(PINECONE_NS or "", {}).get("vector_count", 0)
    return {
        "namespace": PINECONE_NS,
        "title_count": count({"type": "title"}),
        "summary_count": count({"type": "summary"}),
        "all_count": count(None),
        "auto_synonyms_loaded": len(load_auto_syns()),
    }

@app.get("/debug/peek")
def debug_peek(q: str):
    vs = get_vectorstore()
    def sim_with_scores(filter_):
        pairs = vs.similarity_search_with_score(q, k=5, filter=_pc_filter(filter_))
        return [
            {
                "score": float(sc),
                "type": (d.metadata or {}).get("type"),
                "doc_id": (d.metadata or {}).get("doc_id"),
                "title": (d.metadata or {}).get("title") or (d.page_content or "")[:100],
                "link": (d.metadata or {}).get("link"),
            }
            for d, sc in pairs
        ]
    return {
        "q": q,
        "title_only": sim_with_scores({"type": "title"}),
        "summary_only": sim_with_scores({"type": "summary"}),
        "mixed": sim_with_scores(None),
    }

# 새 출력 포맷(프론트 규격)에 맞춰 response_model 미사용
@app.post("/chat")
def chat(turn: Dict[str, Any]):
    # 1) 대화 전체 맥락을 질의로 구성 (HyDE 허용 여부 동시 반환)
    conv_q, allow_hyde = build_conversational_query(turn)
    conv_q = (conv_q or "").strip()
    if not conv_q:
        return {"found": False, "assistant": "무엇을 찾고 계신가요? 키워드를 한두 개 알려주세요."}

    # 2) 동의어 확장 + (조건부) HyDE
    query = make_query(conv_q, allow_hyde=allow_hyde)
    logging.info("[CONV] raw=%r | final_query=%r | allow_hyde=%s", conv_q, query, allow_hyde)

    # 3) 타이틀 우선 검색
    title_hits = _title_first(query, k=TOP_K, fetch_k=FETCH_K)
    reps = group_by_doc(title_hits)

    # (폴백) 타이틀 비면 무필터 혼합으로 후보 확보
    if not reps:
        mixed_docs = _mmr_docs(query, k=TOP_K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MMR, flt=None)
        mixed_scored = _score_docs(query, mixed_docs)
        boosted = []
        for d, sc in mixed_scored:
            md = d.metadata or {}
            title = md.get("title") or d.page_content or ""
            sc2 = sc
            sc2 += _title_token_overlap_boost(query, title, TITLE_BOOST)
            sc2 += _lexical_token_boost(query, title, per_token_partial=0.25, per_token_exact=0.35)
            boosted.append((d, sc2))
        boosted.sort(key=lambda t: t[1], reverse=True)
        reps = group_by_doc(boosted)

    # 4) 베스트 선택 (타이틀 강하면 채택, 아니면 요약 보완)
    if reps and reps[0][1] >= COS_THRESHOLD:
        best_doc_id, best_score, best_pack = reps[0]
    else:
        summary_hits = _summary_mmr(query, k=TOP_K, fetch_k=FETCH_K)
        reps2 = group_by_doc(summary_hits)
        combined: Dict[str, Tuple[str, float, Dict[str, Any]]] = {}
        for doc_id, sc, pack in reps + reps2:
            if (doc_id not in combined) or (sc > combined[doc_id][1]):
                combined[doc_id] = (doc_id, sc, pack)
        if not combined:
            return {
                "found": False,
                "assistant": "관련 공지를 찾지 못했어요. 찾으시는 공지에 대해 더 알려주세요!"
            }
        merged = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        best_doc_id, best_score, best_pack = merged[0]
        reps = merged

    # 5) 동일 doc_id의 summary 청크를 추가로 합쳐 컨텍스트 강화
    best_pack = enrich_pack_with_summaries(best_pack, query, top_chunks=12)

    # refs 상위 3개 (id 포함)
    refs: List[Dict[str, Any]] = []
    for i, (doc_id, s, p) in enumerate(reps[:3], start=1):
        m = doc_meta(p)
        refs.append({"id": m["id"], "title": m["title"], "link": m["link"], "score": s})

    # 6) Threshold 미달 → 두 개만 반환
    if best_score < COS_THRESHOLD:
        return {
            "found": False,
            "assistant": "조금 모호해요. 찾으시는 공지에 대해 더 알려주세요!"
        }

    # 7) 답변 생성
    ctx = build_context(best_pack, max_chars=1800)
    m = doc_meta(best_pack)
    prompt = (
        f"[사용자 대화 요약 질의]\n{conv_q}\n\n"
        f"[선정 문서 메타]\n제목: {m['title']}\n게시일: {m['posted_date']}\n부서: {m['department']}\n"
        f"[컨텍스트]\n{ctx}\n"
    )
    assistant_message = get_llm().invoke([("system", SYSTEM_FINAL), ("user", prompt)]).content.strip()

    selected = _make_selected_with_summary(best_pack, max_chars=400)

    # 최종 새 포맷
    return {
        "found": True,
        "assistant": assistant_message,
        "selected": selected
    }
