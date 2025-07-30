from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from fastapi import UploadFile


# 请求问卷模型
class ChatRequest(BaseModel):
    user_ip: str
    message: str
    user_data: dict

# 带文件的聊天请求模型（用于Form数据）
class ChatWithFileRequest(BaseModel):
    user_ip: str
    message: str
    user_data: str  # JSON字符串格式
    file_description: Optional[str] = Field(default="", description="文件描述")

# 文件上传请求模型
class FileUploadRequest(BaseModel):
    user_ip: str
    file_type: str = Field(description="文件类型：image, pdf, report等")
    description: Optional[str] = Field(default="", description="文件描述")

# 文件上传响应模型
class FileUploadResponse(BaseModel):
    user_ip: str
    file_id: str
    file_path: str
    extracted_content: str
    message: str

# ai回复响应模型
class ChatResponse(BaseModel):
    user_ip: str
    ai_reply: str

#请求预制词模型
class PresetWordsRequest(BaseModel):
    user_ip: str

# 生成预制词
class PresetWordsResponse(BaseModel):
    preset_words: list