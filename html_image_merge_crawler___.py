# html_image_merge_crawler_lite.py
import os, re, json, time, base64, traceback, requests
from io import BytesIO
from typing import Optional, Dict, List
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv; load_dotenv()

from bs4 import BeautifulSoup
from PIL import Image
import mysql.connector
from mysql.connector import Error as MySQLError
from openai import OpenAI

# Optional Playwright
try:
    from playwright.sync_api import sync_playwright
    _PW = True
except Exception:
    _PW = False

# ===== Config =====
BASE_URL  = "https://www.uos.ac.kr/korNotice/view.do"
LIST_URL  = "https://www.uos.ac.kr/korNotice/list.do"
OUT_DIR   = os.path.join(os.path.abspath(os.getcwd()), "notices_img"); os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES: Dict[str, str] = {
    "COLLEGE_ENGINEERING": "20013DA1",
    "COLLEGE_HUMANITIES": "human01",
    "COLLEGE_SOCIAL_SCIENCES": "econo01",
    "COLLEGE_URBAN_SCIENCE": "urbansciences01",
    "COLLEGE_ARTS_SPORTS": "artandsport01",
    "COLLEGE_BUSINESS": "20008N2",
    "COLLEGE_NATURAL_SCIENCES": "scien01",
    "COLLEGE_LIBERAL_CONVERGENCE": "clacds01",
    "GENERAL": "FA1",
    "ACADEMIC": "FA2",
}
CATEGORY_LIST_PARAMS: Dict[str, Dict[str, str]] = {
    "COLLEGE_ENGINEERING": {"cate_id2": "000010383"},
}

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUMMARIZE_MODEL = "gpt-4o"
EMBED_MODEL     = "text-embedding-3-small"

REQUEST_SLEEP=1.0
PLAYWRIGHT_TIMEOUT_MS=45000
RECENT_WINDOW=100

# ===== DB Utils =====
@contextmanager
def mysql_conn():
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback(); raise
    finally:
        conn.close()

UPSERT_SQL = """
INSERT INTO notice (category, post_number, title, link, summary, embedding_vector, posted_date, department)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s) AS new
ON DUPLICATE KEY UPDATE
title=new.title, link=new.link, summary=new.summary, embedding_vector=new.embedding_vector,
posted_date=new.posted_date, department=new.department
"""
EXISTS_SQL = "SELECT 1 FROM notice WHERE category=%s AND post_number=%s AND posted_date=%s LIMIT 1"

def upsert_notice(row: dict):
    with mysql_conn() as conn:
        conn.cursor().execute(UPSERT_SQL, (
            row["category"], row["post_number"], row["title"], row["link"],
            row.get("summary"), row.get("embedding_vector"),
            row["posted_date"], row.get("department"),
        ))

def exists_notice(category: str, post_number: int, posted_date: Optional[str]) -> bool:
    with mysql_conn() as conn:
        cur = conn.cursor(); cur.execute(EXISTS_SQL, (category, post_number, posted_date))
        r = cur.fetchone(); cur.close(); return r is not None

# ===== HTTP / Parse =====
def fetch(url: str, params: dict) -> Optional[str]:
    try:
        r = requests.get(url, params=params, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        return r.text if r.status_code==200 else None
    except: return None

def parse_date_yyyy_mm_dd(text: str) -> Optional[str]:
    m = re.search(r"\d{4}-\d{2}-\d{2}", text or ""); return m.group(0) if m else None

def extract_main_text(html: str, max_chars=12000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = next((soup.select_one(sel) for sel in
                 ["div.vw-cnt","div.vw-con","div.vw-bd","div.board-view","article","div#content","div#contents","main"]
                 if soup.select_one(sel) and soup.select_one(sel).get_text(strip=True)), soup.body or soup)
    for ks in [".related",".relate",".attach",".file",".files",".prev",".next","footer","#footer",".sns",".share",".copyright",".copy",".address",".addr"]:
        [n.decompose() for n in main.select(ks)]
    text = re.sub(r"(서울시립대학교\s*.+?\d{2,3}-\d{3,4}-\d{4}|Copyright.+?All rights reserved\.?|이전글.*|다음글.*|관련\s?게시물.*)",
                  "", main.get_text("\n", strip=True), flags=re.I|re.S)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars] + ("\n\n[... 본문 일부 생략 ...]" if len(text)>max_chars else "")

def parse_notice_fields(html: str, seq: int) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    t = soup.select_one("div.vw-tibx h4")
    if not t: return None
    spans = soup.select("div.vw-tibx div.zl-bx div.da span")
    department = spans[1].get_text(strip=True) if len(spans)>=3 else ""
    dt = parse_date_yyyy_mm_dd(spans[2].get_text(strip=True) if len(spans)>=3 else "") or datetime.now().strftime("%Y-%m-%d")
    post_number_el = soup.select_one("input[name=seq]")
    post_number = int(post_number_el["value"]) if post_number_el and post_number_el.get("value") else int(seq)
    return {"title": t.get_text(strip=True), "department": department, "posted_date": dt, "post_number": post_number}

# ===== Playwright =====
def html_to_images(url: str, viewport_w=1200, slice_h=1800, timeout_ms=PLAYWRIGHT_TIMEOUT_MS,
                   debug_full_path: Optional[str]=None) -> List[Image.Image]:
    if not _PW: return []
    try:
        with sync_playwright() as p:
            page = p.chromium.launch(headless=True, args=["--disable-web-security","--hide-scrollbars"]).new_page(
                viewport={"width":viewport_w,"height":slice_h}, device_scale_factor=2.0)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            for _ in range(6):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)"); page.wait_for_timeout(500)
            page.wait_for_load_state("networkidle", timeout=timeout_ms); page.wait_for_timeout(500)
            buf = page.screenshot(full_page=True, type="png"); page.context.browser.close()
        if debug_full_path:
            open(debug_full_path,"wb").write(buf)
        full = Image.open(BytesIO(buf)).convert("RGB"); W,H = full.size; y=0; imgs=[]
        while y<H:
            imgs.append(full.crop((0,y,W,min(y+slice_h,H)))); y+=slice_h
        return imgs
    except: return []

def pil_to_data_url(img: Image.Image) -> str:
    b=BytesIO(); img.save(b, format="PNG", optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode("utf-8")

# ===== OpenAI =====
def summarize(html_text: str, images: List[Image.Image]) -> str:
    contents=[{"type":"input_text","text":(
        "아래는 대학 공지 'HTML 본문 텍스트'입니다. 이 텍스트를 우선 근거로 하고, 이어지는 전체 캡처 이미지들에만 있는 "
        "표/포스터/스캔 문장 정보를 보완해 자연어 문단으로 상세 요약하세요. 본문과 무관한 사이드/푸터/주소/관련글은 제외.\n"
        "- 수치/날짜/시간/기관/장소/연락처는 원문 표기 그대로 사용.\n\n[HTML]\n"+html_text)}]
    contents += [{"type":"input_image","image_url":pil_to_data_url(im)} for im in images]
    try:
        r = client.responses.create(model=SUMMARIZE_MODEL, input=[{"role":"user","content":contents}], temperature=0.2)
        return (r.output_text or "").strip()
    except Exception as e:
        print(f"summary err: {e}"); return ""

def embed(text: str) -> Optional[str]:
    if not text: return None
    try:
        v = client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding
        return json.dumps(v)
    except Exception as e:
        print(f"embed err: {e}"); return None

# ===== List seq extract (고정글 제외 포함 통합) =====
_SEQ_PATTERNS = [
    r"view\.do[^\"'>]*(?:\?|&|&amp;)seq=(\d+)",
    r"\(\s*['\"][^'\"]*['\"]\s*,\s*'(\d+)'\s*\)",
    r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)",
]
def extract_seqs(html: str, skip_pinned=True) -> List[int]:
    soup = BeautifulSoup(html, "html.parser")
    seqs=[]
    if skip_pinned:
        for li in soup.select("li"):
            num = li.select_one("p.num")
            if num and (num.select_one("span.cl") or "공지" in num.get_text(strip=True)): continue
            chunk = li.decode()
            found=None
            for pat in _SEQ_PATTERNS:
                m=re.search(pat, chunk)
                if m: found=int(m.group(1)); break
            if found: seqs.append(found)
    else:
        txt=str(soup)
        for pat in _SEQ_PATTERNS:
            seqs += [int(m.group(1)) for m in re.finditer(pat, txt)]
    # order-preserving unique
    seen=set(); out=[]
    for s in seqs:
        if s not in seen: seen.add(s); out.append(s)
    return out

def collect_recent_seqs(list_id: str, extra: Optional[Dict[str,str]]=None, limit=RECENT_WINDOW, max_pages=10)->List[int]:
    collected=[]; seen=set()
    for page in range(1, max_pages+1):
        params={"list_id":list_id,"pageIndex":str(page),"searchCnd":"","searchWrd":""}
        if extra: params.update(extra)
        html = fetch(LIST_URL, params); 
        if not html: break
        page_seqs = extract_seqs(html, skip_pinned=(page==1))
        for s in page_seqs:
            if s not in seen:
                seen.add(s); collected.append(s)
                if len(collected)>=limit: return collected
        if not page_seqs: break
        time.sleep(0.2)
    return collected

# ===== Processing =====
def process_one(category_key: str, list_id: str, seq: int) -> str:
    html = fetch(BASE_URL, {
        "list_id":list_id,"seq":str(seq),"sort":"1","pageIndex":"1","viewAuth":"Y","writeAuth":"Y",
        "board_list_num":"10","lpageCount":"12"
    })
    if not html: return "skipped_error"
    parsed = parse_notice_fields(html, seq)
    if not parsed: return "not_found"
    link = f"{BASE_URL}?list_id={list_id}&seq={seq}"

    if exists_notice(category_key, parsed["post_number"], parsed["posted_date"]): return "stored"
    html_text = extract_main_text(html)
    imgs = html_to_images(link, debug_full_path=os.path.join(OUT_DIR, f"{category_key}_{seq}_FULL.png"))
    if not imgs: return "skipped_error"

    summary = summarize(html_text, imgs)
    if not summary: return "skipped_error"
    emb = embed(summary)

    row = {
        "category": category_key,
        "post_number": parsed["post_number"],
        "title": parsed["title"],
        "link": link,
        "summary": summary,
        "embedding_vector": emb,
        "posted_date": parsed["posted_date"],
        "department": parsed["department"],
    }
    try:
        upsert_notice(row)
        print(f" [{category_key}] seq={seq}, no={parsed['post_number']}, date={parsed['posted_date']}, title={parsed['title'][:30]}...")
        return "stored"
    except MySQLError as e:
        print(f"DB err {getattr(e,'errno',None)}: {e}"); print(traceback.format_exc(limit=2)); return "skipped_error"

# ===== Main =====
if __name__ == "__main__":
    targets = [
        # "COLLEGE_ENGINEERING","COLLEGE_HUMANITIES","COLLEGE_SOCIAL_SCIENCES","COLLEGE_URBAN_SCIENCE",
        # "COLLEGE_ARTS_SPORTS","COLLEGE_BUSINESS","COLLEGE_NATURAL_SCIENCES",
        "COLLEGE_LIBERAL_CONVERGENCE",
    ]
    for cat in targets:
        list_id = CATEGORIES.get(cat)
        if not list_id: 
            print(f"⏭ {cat}: list_id 미설정"); continue
        seqs = collect_recent_seqs(list_id, extra=CATEGORY_LIST_PARAMS.get(cat), limit=RECENT_WINDOW, max_pages=10)
        if not seqs: 
            print(f" {cat}: seq 없음"); continue
        print(f"==== [{cat}] list_id={list_id}, 수집 {len(seqs)}개 ====")
        for seq in reversed(seqs):
            process_one(cat, list_id, seq); time.sleep(REQUEST_SLEEP)

# 중복 함수 통합
# jpeg, png -> png 고정
# png가 용량이 더 큰데(토큰 많이 나옴) 정확도, 안정성이 높음 -> 토큰 너무 많이 나온다 싶으면 jpeg로 변경
# 호출부 간소화