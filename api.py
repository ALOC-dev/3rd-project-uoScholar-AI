# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI  # 최신 OpenAI Chat 모델

# 환경변수 로드 (.env 파일 또는 시스템 환경변수에서 API 키 불러오기)
load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"  # 환경변수에 설정되었으면 생략 가능

# GPT 모델 객체 생성 (gpt-4o 사용)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 사용자 질문 입력
user_input = input("질문을 입력하세요: ").strip()

# GPT에게 질문에 맞는 검색 키워드를 추출하라고 지시
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

# GPT 호출
response = llm.invoke(prompt)

# 응답 출력
print("\n🧠 GPT가 추출한 키워드:")
print(response.content.strip())
