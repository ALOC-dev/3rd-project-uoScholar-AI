# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI  # 최신 OpenAI Chat 모델

# 환경변수 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] =

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 사용자 질문 입력
user_input = input("질문을 입력하세요: ").strip()

#프롬포팅
prompt = f"""
너는 사용자의 자연어 질문을 이해하고, 다음 세 가지 항목으로 나누어 키워드를 JSON 형식으로 정리해야 해.

1. "title_keywords": 공지 제목에 들어갈 만한 핵심 단어들
2. "writer_keywords": 공지를 작성했을 가능성이 있는 부서명
3. "year": 특정 학년도나 연도가 명시된 경우 (숫자), 없으면 null

[사용자 질문]
"{user_input}"

[출력 형식 예시]
{{
  "title_keywords": ["딥러닝", "수업"],
  "writer_keywords": ["인공지능학부"],
  "year": 2024
}}

형식에 맞춰 JSON만 출력해줘.
"""

response = llm.invoke(prompt)

# 출력
print("\n")
print(response.content.strip())
