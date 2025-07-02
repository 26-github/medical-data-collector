from pydantic import BaseModel
from fastapi import FastAPI



app = FastAPI()

# 请求问卷模型
class ChatRequest(BaseModel):
    user_id: int
    message: str

# ai回复响应模型
class ChatResponse(BaseModel):
    user_id: int
    ai_reply: str
    user_data: dict

#请求预制词模型
class PresetWordsRequest(BaseModel):
    user_id: int

# 生成预制词
class PresetWordsResponse(BaseModel):
    preset_words: list