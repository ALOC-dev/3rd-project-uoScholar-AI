import requests
from bs4 import BeautifulSoup
import re
import time

base_url = "https://www.uos.ac.kr/korNotice/view.do"
headers = {
    'User-Agent': 'Mozilla/5.0'
}

# 수집할 seq 범위
start_seq = 14773
end_seq = 15385

hashtags = set()

for seq in range(start_seq, end_seq + 1):
    params = {
        'list_id': '20013DA1',
        'seq': seq,
        'sort': '1',
        'pageIndex': '1',
        'searchCnd': '1',
        'searchWrd': '',
        'cate_id': '',
        'viewAuth': 'Y',
        'writeAuth': 'Y',
        'board_list_num': '10',
        'lpageCount': '12',
        'menuid': ''
    }

    try:
        res = requests.get(base_url, params=params, headers=headers)
        if res.status_code != 200:
            continue

        soup = BeautifulSoup(res.text, 'html.parser')
        body_text = soup.get_text()

        found = re.findall(r'#\w+', body_text)
        hashtags.update(found)

        print(f"[{seq}] found: {found}")
    except Exception as e:
        print(f"[{seq}] error: {e}")

print("\n📦 최종 수집된 해시태그:")
print(sorted(hashtags))
