import requests
from bs4 import BeautifulSoup
import re
import time

base_url = "https://www.uos.ac.kr/korNotice/view.do"
headers = {
    'User-Agent': 'Mozilla/5.0'
}

start_seq = 15300
end_seq = 15350

go_detail_values = set()

for seq in range(start_seq, end_seq + 1):
    params = {
        'list_id': '20013DA1',
        'seq': seq,
        'sort': '1',
        'pageIndex': '1',
        'searchCnd': '',
        'searchWrd': '',
        'cate_id': '',
        'viewAuth': 'Y',
        'writeAuth': 'Y',
        'board_list_num': '10',
        'lpageCount': '12',
        'menuid': ''
    }

    try:
        res = requests.get(base_url, headers=headers, params=params, timeout=5)
        if res.status_code != 200:
            print(f"[{seq}] 접속 실패 (status {res.status_code})")
            continue

        soup = BeautifulSoup(res.text, 'html.parser')
        a_tags = soup.find_all('a', onclick=True)

        found = []
        for tag in a_tags:
            onclick = tag.get('onclick', '')
            match = re.search(r"goMoreDetail\('([^']+)'\)", onclick)
            if match:
                value = match.group(1).strip()
                go_detail_values.add(value)
                found.append(value)

        if found:
            print(f"[{seq}] ✅ 추출된 항목: {found}")
        else:
            print(f"[{seq}] 없음")

        time.sleep(0.3)  # 서버에 부담 주지 않도록 딜레이

    except Exception as e:
        print(f"[{seq}] ❌ 오류 발생: {e}")

print("\n📦 최종 수집 결과:")
for item in sorted(go_detail_values):
    print(f"- {item}")
