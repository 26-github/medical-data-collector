from pydantic import BaseModel, Field
from typing import List, Dict, Any


# 请求问卷模型
class ChatRequest(BaseModel):
    user_ip: str
    message: str
    language:str

# ai回复响应模型
class ChatResponse(BaseModel):
    user_ip: str
    ai_reply: str
    user_data: dict

#请求预制词模型
class PresetWordsRequest(BaseModel):
    user_ip: str

# 生成预制词
class PresetWordsResponse(BaseModel):
    preset_words: list