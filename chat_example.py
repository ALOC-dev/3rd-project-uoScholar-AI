# convo_api.py
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, time

from openai import OpenAI

# 네가 만든 모듈
from run_retriever import MySQLHybridRetriever, embed_fn, DB_CONFIG

# ====== 전역 준비 ======
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
COS_THRESHOLD = float(os.getenv("COS_THRESHOLD", "0.6"))  # 임계치
MIN_USER_TURNS_FOR_FINAL = int(os.getenv("MIN_USER_TURNS_FOR_FINAL", "2"))  # 최종답변 전 최소 사용자 발화 수

retriever = MySQLHybridRetriever(
    conn_uoscholar=DB_CONFIG,
    embed_fn=embed_fn,
    k=5,
    candidate_n=200,
)

# 세션 상태(데모용 인메모리; 실서비스는 Redis/DB로 수명관리)
SESS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SEC = 3600  # 1시간


# ====== 스키마 ======
class ChatTurn(BaseModel):
    session_id: str
    user_message: str

class ChatReply(BaseModel):
    found: bool                # 임계치 충족 여부 (확정 플래그)
    assistant_message: str     # LLM이 사용자에게 보낼 메시지(추가 질문 or 최종 설명)
    top_score: float           # 현재 라운드 최상단 코사인 유사도(진단용)
    selected: Optional[Dict[str, Any]] = None  # found=True일 때 선택된 공지 메타
    refs: Optional[List[Dict[str, Any]]] = None  # 컨텍스트로 사용된 상위 문서(간단)


# ====== 유틸 ======
def _get_or_create_session(sid: str) -> Dict[str, Any]:
    now = time.time()
    sess = SESS.get(sid)
    if sess is None or (now - sess.get("ts", now)) > SESSION_TTL_SEC:
        sess = {"queries": [], "ts": now}
        SESS[sid] = sess
    else:
        sess["ts"] = now
    return sess

def _merge_queries(queries: List[str], max_len: int = 256) -> str:
    # 간단히 공백으로 이어주고 너무 길면 자르기
    q = " ".join([q.strip() for q in queries if q.strip()])
    return q[:max_len]

SYSTEM_FOLLOWUP = """
당신은 친절하고 효율적인 대학 공지 검색 도우미 "공지봇"입니다.
당신의 임무는 사용자가 원하는 공지를 더 정확히 찾을 수 있도록 최소한의 핵심 정보를 질문하는 것입니다.

**역할:**
- 사용자가 처음 던진 질문만으로는 정확한 공지를 찾기 어려운 경우,
  필요한 보충 정보를 단 한 문장으로 질문합니다.
- 질문은 친절하고 존댓말로 작성하며, 불필요한 설명은 포함하지 않습니다.


** 주의사항:**
- 앞서 언급한 '고려 해야할 정보들'을 있는 그대로 묻지 마세요. 당신이 사용자의 기존 질문들에 대해 사용자의 질문 의도를 파악하고, 이를 응용해서 질문해야합니다. 
- 예시
    * 사용자 : "혹시 지금 교내 근로장학생 모집하고 있어?"
    * LLM(당신) : "근로장학생을 신청하고 싶군요!😊 혹시 어떤 내용의 근무에 관심 있으신가요?"
   - 자연스럽게 정보를 수집하되, 인터뷰 같지 않게 하기

**대화 스타일:**
    - 친근하고 격려적인 톤, 적절한 이모지 사용
    - 한 번에 너무 많은 질문 하지 않기 (1-2개만)
    - 사용자의 답변에 공감하고 간단한 질문으로 이어가기
    - 자연스럽게 정보를 수집하되, 인터뷰 같지 않게 하기
"""

def gen_followup_question(user_summary: str) -> str:
    msg = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_FOLLOWUP},
            {"role": "user", "content": f"사용자 요구 요약: {user_summary}\n최소한의 추가 확인 질문을 한 문장으로만 만들어 주세요."}
        ],
    )
    return msg.choices[0].message.content.strip()

# 자연어 한 문단 답변 전용 시스템 프롬프트
SYSTEM_FINAL = """
당신은 대학 공지사항 안내 도우미 "공지봇"입니다.
당신의 임무는 검색된 공지의 메타데이터와 summary를 바탕으로 사용자가 궁금해하는 질문에 대한 답변을 생성하는 것입니다.

**답변 작성 규칙:**
1. 🎯 제공된 데이터(summary, 메타데이터)만 근거로 사용한다.
2. 📊 핵심 항목을 구조적으로 안내한다:
   - 제목
   - 일정/기간
   - 대상
   - 절차/방법
   - 담당 부서/문의
   - 링크
3. ❗ 없는 정보는 "미기재"라고 명시한다.
4. ❌ 추측하거나 summary에 없는 사실은 추가하지 않는다.

**스타일:**
- 존댓말, 간결한 문체
- 3~4문장 이내 요약
- 중복 없이 핵심만 전달
"""


def compose_final_answer(q_text: str, notice: Dict[str, Any]) -> str:
    # notice: {"title","link","summary","posted_date","department","category"}
    title = notice.get("title") or ""
    link  = notice.get("link")  or ""
    summary = notice.get("summary") or ""
    posted = notice.get("posted_date") or ""
    dept   = notice.get("department") or ""
    cat    = notice.get("category") or ""

    # 사용자에게는 "궁극적 질문"을 노출하지 않고, 자연어 한 문단만 출력하도록 지시
    user_prompt = (
        f"대화 기록(누적 질의): {q_text}\n\n"
        f"[공지 메타]\n- 제목: {title}\n- 게시일: {posted}\n- 부서/학과: {dept}\n- 카테고리: {cat}\n- 링크: {link}\n\n"
        f"[공지 summary]\n{summary[:2000]}\n\n"
        "지금까지의 대화를 분석해 사용자가 궁극적으로 묻는 최종 질문을 **내부적으로만 파악**하세요. "
        "출력 시에는 그 질문을 다시 쓰지 말고, 위 summary와 메타데이터만 근거로 "
        "사용자가 원하는 정보를 자연스럽고 간결한 한국어 한 문단으로만 작성하세요. "
        "절대 항목별 나열 형식(예: 제목:, 일정:, 대상:)을 사용하지 마세요."
    )

    msg = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_FINAL},
            {"role": "user", "content": user_prompt},
        ],
    )
    return msg.choices[0].message.content.strip()


# ====== 메인 엔드포인트 ======
app = FastAPI()

@app.post("/chat", response_model=ChatReply)
def chat(turn: ChatTurn):
    sess = _get_or_create_session(turn.session_id)
    # 누적 질의에 이번 입력 추가
    sess["queries"].append(turn.user_message)

    # 누적 질의를 간단 결합(필요하면 최근 N개만)
    combined = _merge_queries(sess["queries"], max_len=256)

    # retriever 호출
    docs = retriever.invoke(combined)  # List[Document], page_content=summary, metadata에 score 포함(코사인)
    if not docs:
        # 후보가 전혀 없으면 추가 질문으로 유도
        q = gen_followup_question(turn.user_message)
        return ChatReply(
            found=False,
            assistant_message=q,
            top_score=0.0,
            refs=[]
        )

    # 최상단 문서와 점수
    top = docs[0]
    top_meta = top.metadata
    top_score = float(top_meta.get("score") or 0.0)

    # 진단용 refs(상위 3개)
    refs = []
    for d in docs[:3]:
        md = d.metadata
        refs.append({
            "id": md.get("id"),
            "title": md.get("title"),
            "link": md.get("link"),
            "posted_date": md.get("posted_date"),
            "category": md.get("category"),
            "department": md.get("department"),
            "score": float(md.get("score") or 0.0),
        })

    # ✅ 최소 턴 수 충족 여부 (사용자 발화 수 기준)
    has_min_turns = len(sess["queries"]) >= MIN_USER_TURNS_FOR_FINAL

    if top_score >= COS_THRESHOLD and has_min_turns:
        # 확정: found=True + 최종 안내 생성
        selected = {
            "id": top_meta.get("id"),
            "title": top_meta.get("title"),
            "link": top_meta.get("link"),
            "posted_date": top_meta.get("posted_date"),
            "category": top_meta.get("category"),
            "department": top_meta.get("department"),
            "score": top_score,
            "summary": top.page_content or "",
        }
        final_answer = compose_final_answer(combined, selected)
        return ChatReply(
            found=True,
            assistant_message=final_answer,
            top_score=top_score,
            selected=selected,
            refs=refs
        )
    else:
        # ❗ 임계치 미만이거나 / 임계치 만족해도 최소 턴 미충족 → 추가 질문 1문장
        user_summary = f"누적 질의: {combined}"
        followup = gen_followup_question(user_summary)
        return ChatReply(
            found=False,
            assistant_message=followup,
            top_score=top_score,
            refs=refs
        )
