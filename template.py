from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(model = "gpt-4o", temperature= 0.2)

user_input = input("질문을 입력해주세요").strip()

prompt =f'''
사용자의 질문에서 중요하다고 생각하는 부분을 2개만 뽑아봐봐

{user_input}

'''

response = llm.invoke(prompt)

print(response.content.strip())