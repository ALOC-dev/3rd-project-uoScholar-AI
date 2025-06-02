from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from notice_search import search_similar_notices

app = FastAPI()

# 요청 형식
class UserQuery(BaseModel):
    user_input: str

# POST 요청 핸들러
@app.post("/search")
def search_notices(query: UserQuery):
    try:
        results = search_similar_notices(query.user_input)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))