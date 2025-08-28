# build_auto_synonyms.py
import os, re, json, mysql.connector, time
from dotenv import load_dotenv; load_dotenv()
from collections import Counter
from math import sqrt
from langchain_openai import OpenAIEmbeddings

# ===== ENV =====
DB = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": "utf8mb4",
    "connection_timeout": 10,
    "use_pure": True,
    "raise_on_warnings": True,
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
AUTO_SYNS_PATH = os.getenv("AUTO_SYNS_PATH", "data/auto_synonyms.json")

# 튜닝 파라미터(환경변수로 덮어쓰기 가능)
TOPK      = int(os.getenv("AUTO_SYNS_TOPK", "5"))      # 단어당 확장 개수
MIN_FREQ  = int(os.getenv("AUTO_SYNS_MIN_FREQ", "2"))  # 최소 등장 횟수
BATCH_SZ  = int(os.getenv("AUTO_SYNS_BATCH", "256"))

STOP = {
    "공지","안내","모집","프로그램","워크숍","행사","공지사항","문의","신청",
    "대상","참여","학생","모집안내","일정","기간","운영","실시","관련","안내문"
}
TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")

def fetch_titles():
    sql = "SELECT title FROM notice WHERE title IS NOT NULL AND title<>''"
    con = mysql.connector.connect(**DB)
    cur = con.cursor()
    cur.execute(sql)
    rows = [r[0] for r in cur.fetchall()]
    cur.close(); con.close()
    return rows

def tokenize(s: str):
    return [t.lower() for t in TOKEN.findall(s or "") if t]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)) or 1e-9
    nb = sqrt(sum(y*y for y in b)) or 1e-9
    return dot / (na * nb)

def main(topk=TOPK, min_freq=MIN_FREQ):
    assert OPENAI_API_KEY, "OPENAI_API_KEY 필요"
    titles = fetch_titles()
    vocab = []
    for t in titles:
        vocab.extend([w for w in tokenize(t) if w not in STOP and len(w) >= 2])
    freq = Counter(vocab)
    terms = [w for w, c in freq.items() if c >= min_freq]
    print(f"[build] unique terms: {len(terms)} (min_freq={min_freq})")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vecs = {}
    for i in range(0, len(terms), BATCH_SZ):
        batch = terms[i:i+BATCH_SZ]
        vs = embeddings.embed_documents(batch)
        vecs.update({w: v for w, v in zip(batch, vs)})
        print(f"[embed] {i+len(batch)}/{len(terms)}")

    synmap = {}
    # O(n^2) — terms가 매우 많으면 ANN(FAISS 등)로 교체 권장
    for idx, w in enumerate(terms, 1):
        v = vecs[w]
        sims = []
        for u in terms:
            if u == w: continue
            sims.append((u, cosine(v, vecs[u])))
        sims.sort(key=lambda x: x[1], reverse=True)
        synmap[w] = [u for u, _ in sims[:topk]]
        if idx % 200 == 0:
            print(f"[pairwise] {idx}/{len(terms)}")

    payload = {
        "created_at": int(time.time()),
        "model": EMBED_MODEL,
        "min_freq": min_freq,
        "topk": topk,
        "terms": terms,
        "synonyms": synmap,
    }
    os.makedirs(os.path.dirname(AUTO_SYNS_PATH) or ".", exist_ok=True)
    with open(AUTO_SYNS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[done] saved -> {AUTO_SYNS_PATH}")

if __name__ == "__main__":
    main()
