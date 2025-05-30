import pandas as pd
import json
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

# 1. 환경 변수 및 API 키 설정
os.environ["OPENAI_API_KEY"] = "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"
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. 임베딩 함수
def embed_text(text: str) -> list:
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 실패: {text}")
        print("에러:\n", e)
        return [0.0] * 1536  # 실패 시 기본 벡터

# 3. 데이터 로드
df = pd.read_excel("c:/Users/김동환/OneDrive/바탕 화면/3학년 파일/3rd-project-uoScholar-AI/dataset.xlsx")

data = []

# 4. 각 질문-답변 쌍 임베딩 및 저장
for idx, row in df.iterrows():
    question = row['question']
    answer = row['answer']

    print(f"[{idx+1}/{len(df)}] 임베딩 중...")

    q_embed = embed_text(question)
    time.sleep(0.5)  

    a_embed = embed_text(answer)
    time.sleep(0.5)

    data.append({
        "question": question,
        "answer": answer,
        "q_embed": q_embed,
        "a_embed": a_embed
    })

# 5. JSON 파일로 저장
with open("embedded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n 임베딩 완료")
