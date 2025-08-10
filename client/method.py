import os
import time

import httpx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import logging
from mcp import ClientSession

# 增强的全局缓存系统
_global_cache = {}
_rag_cache = {}  # RAG查询结果缓存
_language_cache = {}  # 语言检测缓存
_response_cache = {}  # 完整响应缓存

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent-Server")

def count_tokens(text: str) -> int:
    """
    计算文本的token数量
    对于阿里云DashScope模型，优先使用更准确的计算方法
    """
    if not text:
        return 0
    
    text = str(text)
    
    # 方法1: 尝试使用tiktoken（OpenAI标准）作为参考
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tiktoken_count = len(encoding.encode(text))
        # 对于阿里云DashScope模型，根据经验调整系数（阿里云模型token计算可能略有不同）
        adjusted_count = int(tiktoken_count * 1.1)  # 保守估计，增加10%
        return adjusted_count
    except ImportError:
        logger.info("tiktoken库未安装，使用优化的字符数估算。建议安装: pip install tiktoken")
    except Exception as e:
        logger.warning(f"tiktoken计算失败: {e}")
    
    # 方法2: 使用更精确的字符数估算（针对中英文混合文本优化）
    try:
        # 统计中文字符、英文单词、数字和符号
        import re
        
        # 中文字符（包括中文标点）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
        
        # 英文单词（按空格分割）
        english_text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text)
        english_words = len([word for word in english_text.split() if word.strip()])
        
        # 数字和特殊符号
        remaining_chars = len(re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s]', '', text))
        
        # 根据阿里云DashScope模型的token计算规律估算
        # 中文: 1字符 ≈ 1token
        # 英文: 1单词 ≈ 1.3token (考虑subword)
        # 符号: 1字符 ≈ 1token
        estimated_tokens = chinese_chars + int(english_words * 1.3) + remaining_chars
        
        # 添加基础overhead（消息格式等）
        if estimated_tokens > 0:
            estimated_tokens += 10  # 基础格式token
        
        return max(estimated_tokens, 1)  # 至少返回1
        
    except Exception as e:
        logger.warning(f"优化token计算失败: {e}")
        
    # 方法3: 最简单的fallback
    # 保守估计: 平均每2个字符1个token
    return max(len(text) // 2, 1)

def calculate_prompt_tokens(messages: list, user_data: dict = None) -> dict:
    """
    计算提示词的token消耗
    返回详细的token使用统计
    """
    token_stats = {
        "system_prompt_tokens": 0,
        "conversation_history_tokens": 0,
        "user_data_tokens": 0,
        "current_message_tokens": 0,
        "tool_message_tokens": 0,
        "ai_message_tokens": 0,
        "total_input_tokens": 0,
        "message_count": 0
    }
    
    if not messages:
        return token_stats
    
    for i, message in enumerate(messages):
        try:
            content = ""
            message_type = "unknown"
            
            # 提取消息内容和类型
            if hasattr(message, 'content') and hasattr(message, '__class__'):
                content = str(message.content) if message.content else ""
                message_type = message.__class__.__name__
            elif isinstance(message, dict):
                content = str(message.get('content', ''))
                message_type = message.get('type', message.get('role', 'unknown'))
            else:
                content = str(message)
                
            # 计算token数量
            tokens = count_tokens(content)
            token_stats["message_count"] += 1
            
            # 根据消息类型分类统计
            if isinstance(message, SystemMessage) or message_type in ['system', 'SystemMessage']:
                token_stats["system_prompt_tokens"] += tokens
            elif isinstance(message, HumanMessage) or message_type in ['human', 'user', 'HumanMessage']:
                # 最后一条用户消息作为当前消息
                if i == len(messages) - 1 or not any(
                    isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get('type') in ['human', 'user'])
                    for m in messages[i+1:]
                ):
                    token_stats["current_message_tokens"] = tokens
                token_stats["conversation_history_tokens"] += tokens
            elif isinstance(message, AIMessage) or message_type in ['ai', 'assistant', 'AIMessage']:
                token_stats["ai_message_tokens"] += tokens
                token_stats["conversation_history_tokens"] += tokens
            elif isinstance(message, ToolMessage) or message_type in ['tool', 'function', 'ToolMessage']:
                token_stats["tool_message_tokens"] += tokens
                token_stats["conversation_history_tokens"] += tokens
            else:
                # 其他类型消息归入对话历史
                token_stats["conversation_history_tokens"] += tokens
                
        except Exception as e:
            logger.warning(f"处理消息时出错 (索引 {i}): {e}")
            # 对出错的消息使用默认token估算
            fallback_tokens = len(str(message)) // 3  # 保守估计
            token_stats["conversation_history_tokens"] += fallback_tokens
    
    # 计算用户数据的token数量
    if user_data:
        try:
            user_data_text = str(user_data)
            token_stats["user_data_tokens"] = count_tokens(user_data_text)
        except Exception as e:
            logger.warning(f"计算用户数据token时出错: {e}")
            token_stats["user_data_tokens"] = len(str(user_data)) // 3
    
    # 计算总输入token数
    token_stats["total_input_tokens"] = (
        token_stats["system_prompt_tokens"] + 
        token_stats["conversation_history_tokens"] + 
        token_stats["user_data_tokens"]
    )
    
    # 添加调试信息
    logger.debug(f"Token统计详情: {token_stats}")
    
    return token_stats


def test_token_calculation():
    """
    测试token计算功能
    """
    test_cases = [
        ("Hello world", "简单英文"),
        ("你好世界", "简单中文"),
        ("Hello 你好 world 世界!", "中英文混合"),
        ("", "空字符串"),
        ("这是一个很长的测试文本，包含中文和English mixed content, with numbers 123 and symbols @#$%^&*()", "复杂混合文本"),
    ]
    
    logger.info("🧪 开始测试token计算功能...")
    
    for text, description in test_cases:
        try:
            tokens = count_tokens(text)
            char_count = len(text)
            ratio = tokens / max(char_count, 1)
            logger.info(f"  {description}: '{text[:30]}...' -> {tokens} tokens (字符数: {char_count}, 比例: {ratio:.2f})")
        except Exception as e:
            logger.error(f"  {description}: 计算失败 - {e}")
    
    # 测试消息列表计算
    test_messages = [
        HumanMessage(content="你好，我想咨询一些问题"),
        AIMessage(content="你好！我很乐意帮助您。请问有什么问题？"),
        HumanMessage(content="我想了解阿里云DashScope模型的token计算方式")
    ]
    
    try:
        stats = calculate_prompt_tokens(test_messages)
        logger.info(f"📊 消息列表token统计: {stats}")
    except Exception as e:
        logger.error(f"消息列表token计算失败: {e}")
    
    logger.info("✅ Token计算测试完成")


class data(BaseModel):          #生成用户数据模型
    """用户数据模型，包含要收集的用户信息和AI助手回复"""
    name: str = Field(default="0", description="用户的姓名")
    age: str = Field(default="0", description="用户的年龄")
    phone: str = Field(default="0", description="用户的电话")
    email: str = Field(default="0", description="用户的邮箱")
    country: str = Field(default="0", description="用户的国家")
    language: str = Field(default="English", description="用户的偏好语言")
    firstMedicalOpinion: str = Field(default="0", description="用户的第一医疗意见")
    medicalAttachments: str = Field(default="0", description="用户上传的医疗附件")
    medicalAttachmentAnalysis: str = Field(default="0", description="完整的医疗附件分析结果")
    medicalAttachmentFilename: str = Field(default="0", description="医疗附件文件名")
    preferredCity: str = Field(default="0", description="用户倾向就医的城市")
    preferredHospital: str = Field(default="0", description="用户倾向就医的医院")
    treatmentBudget: str = Field(default="0", description="用户的治疗预算")
    treatmentPreference: str = Field(default="0", description="治疗方案倾向: standard-标准, advanced-先进, clinical-临床")
    urgencyLevel: str = Field(default="0", description="紧急程度: low-低, medium-中, high-高, emergency-紧急")
    consultationType: str = Field(default="0", description="咨询类型: online_only-仅咨询, offline_required-需要线下就医")
    remark: str = Field(default="0", description="用户备注")
    reply: str = Field(default="", description="AI助手的回复内容")

class prefabricated_words(BaseModel):           #生成预制词模型
    """根据ai的回复为用户生成三到五个预制词"""
    words: list[str] = Field(default_factory=list, description="预制词列表，内部包含三到五个预制词")

# 使用 PydanticOutputParser 将 data 模型解析为输出格式
parser_data = PydanticOutputParser(pydantic_object=data)
# 使用 PydanticOutputParser 将 prefabricated_words 模型解析为输出格式
parser_prefabricated_words = PydanticOutputParser(pydantic_object=prefabricated_words)

def get_prompt_for_collecting_data():  #生成医疗科普助手的提示模板
    prompt = PromptTemplate(
        template="""你是一个专业的多语言医疗科普助手，能够流利使用各种语言为用户提供医疗咨询服务。

用户查询：{query}

### 语言处理规则：
1. **严格使用用户偏好语言回复**：{language}
2. **小语种支持**：如果用户使用小语种，请确保：
   - 使用准确的语法和表达
   - 采用当地常用的医疗术语
   - 考虑文化背景和语言习惯
   - 保持专业性和准确性
3. **语言代码对应**：
   - zh: 中文（简体/繁体）
   - English: 英文
   - ja: 日文  
   - ko: 韩文
   - ar: 阿拉伯文
   - th: 泰文
   - vi: 越南文
   - hi: 印地文
   - bn: 孟加拉文
   - ta: 泰米尔文
   - te: 泰卢固文
   - ml: 马拉雅拉姆文
   - kn: 卡纳达文
   - gu: 古吉拉特文
   - pa: 旁遮普文
   - ur: 乌尔都文
   - fa: 波斯文
   - ru: 俄文
   - es: 西班牙文
   - fr: 法文
   - de: 德文
   - 其他ISO 639-1语言代码

### 医疗处理规则：
对于医疗相关问题，必须使用medical_qa_search工具获取专业医学知识。

### 用户数据：
{data}

### 对话历史：
{record}

### 回复要求：
1. 使用{language}语言回复
2. 提供专业、准确的医疗信息
3. 保持文化敏感性和语言适应性
4. 如果是小语种，确保医疗术语的准确性

请处理用户的医疗问题。""",
        # query是用户最新消息，data是用户已收集的信息，record是对话记录，language是用户偏好语言
        input_variables=["query", "data", "record","language"],
    )
    return prompt


def extract_reply_from_ai_message(ai_content: str) -> str:
    """从AI消息中提取reply字段，如果是JSON格式的话"""
    try:
        import json
        # 尝试解析JSON
        if ai_content.strip().startswith('{') and ai_content.strip().endswith('}'):
            data = json.loads(ai_content)
            return data.get('reply', ai_content)
    except:
        pass
    return ai_content

def optimize_conversation_history(conversation_history: list, max_entries: int = 4, max_tokens: int = 1500) -> list:
    """优化对话历史，大幅减少token消耗"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    
    if not conversation_history:
        return conversation_history
    
    # 1. 移除工具消息和包含工具调用的AI消息
    cleaned_messages = []
    for msg in conversation_history:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            has_tool_calls = (
                hasattr(msg, 'tool_calls') and msg.tool_calls or
                hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('tool_calls')
            )
            if has_tool_calls:
                continue
        cleaned_messages.append(msg)
    
    # 2. 智能选择最重要的消息
    if len(cleaned_messages) <= max_entries:
        selected_messages = cleaned_messages
    else:
        # 保留最近的消息和包含医疗关键词的重要消息
        important_messages = []
        total_count = len(cleaned_messages)
        preserve_recent = min(2, max_entries // 2)  # 保留最近1-2条消息
        
        # 先添加最近的消息
        for i in range(max(0, total_count - preserve_recent), total_count):
            important_messages.append(cleaned_messages[i])
        
        # 再从历史中选择包含医疗关键词的消息
        medical_keywords = ['症状', '诊断', '治疗', '药物', '检查', '报告', '病情', '医院', '疼痛', '发热']
        for i, msg in enumerate(cleaned_messages[:-preserve_recent]):
            if len(important_messages) >= max_entries:
                break
            if hasattr(msg, 'content'):
                content = str(msg.content).lower()
                if any(keyword in content for keyword in medical_keywords):
                    # 避免重复添加
                    if msg not in important_messages:
                        important_messages.insert(-preserve_recent, msg)
        
        selected_messages = important_messages[:max_entries]
    
    # 3. 激进的内容压缩
    final_messages = []
    total_tokens = 0
    
    for msg in selected_messages:
        if hasattr(msg, 'content'):
            content = str(msg.content)
            # 计算当前token数
            content_tokens = count_tokens(content)
            
            # 如果单条消息过长，进行压缩
            if content_tokens > 300:  # 单条消息最多300 tokens
                # 保留开头和结尾，中间用省略号
                max_chars = 200
                if len(content) > max_chars:
                    start_part = content[:max_chars//2]
                    end_part = content[-max_chars//2:]
                    content = f"{start_part}...[已压缩]...{end_part}"
                    content_tokens = count_tokens(content)
            
            # 检查总token限制
            if total_tokens + content_tokens > max_tokens:
                # 如果会超过限制，进一步压缩或跳过
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # 至少保留50个token的空间
                    # 进一步压缩内容
                    max_chars = min(100, remaining_tokens * 2)  # 粗略估算
                    content = content[:max_chars] + "...[已压缩]"
                    content_tokens = count_tokens(content)
                else:
                    break  # 跳过剩余消息
            
            # 创建压缩后的消息
            if isinstance(msg, HumanMessage):
                final_messages.append(HumanMessage(content=content))
            elif isinstance(msg, AIMessage):
                final_messages.append(AIMessage(content=content))
            else:
                final_messages.append(msg)
            
            total_tokens += content_tokens
        else:
            final_messages.append(msg)
    
    logger.info(f"⚡ 对话历史激进优化: {len(conversation_history)} -> {len(final_messages)} 条消息, 预估tokens: {total_tokens}")
    return final_messages

def clean_conversation_history(conversation_history: list) -> str:
    """彻底清理对话历史，只保留核心对话内容，大幅减少token消耗"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    
    # 使用优化后的历史
    optimized_history = optimize_conversation_history(conversation_history)
    
    history_text = ""
    
    for msg in optimized_history:
        if isinstance(msg, HumanMessage):
            content = str(msg.content)
            history_text += f"用户: {content}\n"
        elif isinstance(msg, AIMessage):
            # 如果content为空但有工具调用，跳过这条消息
            if not msg.content.strip():
                continue
            # 提取AI回复的核心内容
            clean_reply = extract_reply_from_ai_message(msg.content)
            if clean_reply.strip():
                history_text += f"助手: {clean_reply}\n"
        elif isinstance(msg, ToolMessage):
            # 跳过工具消息，因为它们是技术细节，占用大量token
            continue
        elif isinstance(msg, SystemMessage):
            # 跳过系统消息
            continue
    
    return history_text

def build_prompt_text(user_message: str, user_data_dict: dict, conversation_history: list) -> str:
    """构建完整的提示词文本"""
    # 使用新的清理函数获取干净的历史记录
    history_text = clean_conversation_history(conversation_history)

    # 获取原始提示词模板
    prompt_template = get_prompt_for_collecting_data()

    # 构建完整提示词
    full_prompt = prompt_template.format(
        query=user_message,
        data=user_data_dict,
        record=history_text,
        format_instructions=parser_data.get_format_instructions()
    )

    return full_prompt



def get_prompt_for_generating_preset():    #生成预制词的提示模板
    """生成预制词的提示模板，增强小语种支持"""
    prompt = PromptTemplate(
        template="""用户当前数据：{data_and_ai_message}

                 数据说明：
                 - data: 包含用户已收集的信息
                 - reply: AI最新的回复内容
                 - language: 用户偏好语言

                 要求：
                 1. 根据AI的reply内容，生成3-5个预制词选项
                 2. **严格使用language字段指定的语言**生成预制词，支持任何语种包括小语种
                 3. 预制词应该是用户对AI问题的合理回答选项
                 4. 必须包括一个礼貌的拒绝选项，使用对应语言的标准表达：
                    - 中文(zh): "不愿透露"
                    - 英文(English): "Prefer not to say"
                    - 日文(ja): "お答えできません"
                    - 韩文(ko): "답변하고 싶지 않습니다"
                    - 阿拉伯文(ar): "أفضل عدم الإجابة"
                    - 泰文(th): "ไม่ต้องการตอบ"
                    - 越南文(vi): "Không muốn trả lời"
                    - 印地文(hi): "जवाब नहीं देना चाहते"
                    - 孟加拉文(bn): "উত্তর দিতে চাই না"
                    - 泰米尔文(ta): "பதில் சொல்ல விரும்பவில்லை"
                    - 俄文(ru): "Предпочитаю не отвечать"
                    - 西班牙文(es): "Prefiero no decir"
                    - 法文(fr): "Je préfère ne pas dire"
                    - 德文(de): "Möchte ich nicht sagen"
                    - 其他语言：使用相应的礼貌拒绝表达
                 5. 预制词要简短明了，便于用户快速选择
                 6. 智能生成选项（使用对应语言）：
                    - 选择题：提供标准选项
                    - 个人信息：提供"输入"和"跳过"选项  
                    - 年龄：提供年龄段选项
                    - 地理位置：提供常见选项
                    - 预算：提供合理范围选项
                    - 医疗相关：提供常见症状、治疗偏好等选项
                 7. **语言一致性**：所有预制词必须使用传入的language对应的语言进行输出，不得混用
                 8. **小语种特殊处理**：
                    - 确保小语种的语法和表达习惯正确
                    - 使用当地常用的表达方式
                    - 考虑文化背景和语言习惯
                 
                 重要：请直接输出JSON数据，不要输出schema或description！
                 
                 输出示例：
                 {{"words": ["选项1", "选项2", "选项3", "选项4"]}}

                 {format_instructions}""",
        # data_and_ai_message包含用户数据和AI最新回复
        input_variables=["data_and_ai_message"],
        partial_variables={"format_instructions": parser_prefabricated_words.get_format_instructions()},
    )
    return prompt



# 检查用户是否已存在，如果不存在则创建新用户
def check_user(user_ip,database):
    # 如果用户不存在，创建一个新的用户记录
    if user_ip not in database:
        database[user_ip] = {'record': [], 'data': {}}
        return False
    return True

# 获取后续对话的提示模板
def get_follow_up_prompt(token_stats: dict = None):
    """生成后续对话的提示模板，增强多语言支持"""
    
    follow_up_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的多语言医疗科普助手。请根据工具返回的结果提供准确的医疗回复。

## 多语言回复规范：
1. **严格按照用户数据中的language字段使用对应语言回复**
2. **小语种特殊处理**：
   - 确保医疗术语的准确性和专业性
   - 使用当地常用的表达方式
   - 考虑文化背景和语言习惯
   - 保持语法正确和表达自然
3. **支持的语言包括但不限于**：
   - 主要语言：中文(zh)、英文(English)、日文(ja)、韩文(ko)、阿拉伯文(ar)
   - 南亚语言：印地文(hi)、孟加拉文(bn)、泰米尔文(ta)、泰卢固文(te)、马拉雅拉姆文(ml)、卡纳达文(kn)、古吉拉特文(gu)、旁遮普文(pa)、乌尔都文(ur)
   - 东南亚语言：泰文(th)、越南文(vi)、缅甸文(my)、高棉文(km)、老挝文(lo)
   - 中东语言：波斯文(fa)、希伯来文(he)
   - 欧洲语言：俄文(ru)、西班牙文(es)、法文(fr)、德文(de)等
   - 其他小语种：格鲁吉亚文(ka)、亚美尼亚文(hy)、阿姆哈拉文(am)等
4. **回复质量要求**：
   - 医疗信息准确专业
   - 语言表达自然流畅
   - 文化适应性强
   - 用户友好易懂"""),
            MessagesPlaceholder(variable_name="history"),
        ])
    return follow_up_prompt

# 执行工具调用
async def execute_tools(tool_calls: list, session: ClientSession, tools: list) -> list:
    """执行工具调用并生成ToolMessage"""
    tool_messages = []
    for tool_call in tool_calls:
        func_name = "unknown_tool"  # 默认值，确保在异常处理中可用
        try:
            # 修复工具调用格式：OpenAI的工具调用格式
            if "function" in tool_call:
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                # 解析参数字符串为字典
                import json
                args = json.loads(args_str) if args_str else {}
            else:
                # 备用格式（如果是其他格式）
                func_name = tool_call.get("name", "unknown_tool")
                args = tool_call.get("args", {})

            # 查找并执行对应的工具
            tool_found = False
            for t in tools:
                if t.name == func_name:
                    logger.info(f"执行工具: {func_name} 参数: {args}")
                    # 执行MCP工具，传递session参数
                    result = await t.ainvoke(args, session=session)
                    logger.info(f"工具执行结果: {result}")

                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call.get("id", "unknown"),
                            name=func_name,  # 添加name字段，阿里云AI要求function消息必须有name
                        )
                    )
                    tool_found = True
                    break

            if not tool_found:
                logger.warning(f"工具未找到: {func_name}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {func_name} not found",
                        tool_call_id=tool_call.get("id", "unknown"),
                        name=func_name,  # 添加name字段，阿里云AI要求function消息必须有name
                    )
                )
        except Exception as e:
            import traceback
            error_msg = f"工具执行错误: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"完整错误栈: {traceback.format_exc()}")
            tool_messages.append(
                ToolMessage(
                    content=f"Error executing tool: {error_msg}",
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=func_name,  # 添加name字段，阿里云AI要求function消息必须有name
                )
            )
    return tool_messages

# 保留语言检测AI（用户需求）
llm_for_language_detection = ChatOpenAI(
    model="qwen-max",  # 改用更快的qwen-max而不是qwen-vl-max
    temperature=0.1,
    request_timeout=30,  # 缩短超时时间
    max_retries=2,       # 减少重试次数
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
def should_detect_language(user_ip: str, message: str, current_language: str = None) -> bool:
    """智能判断是否需要进行语言检测，增强小语种支持"""
    # 如果消息很短（少于10个字符），但包含特殊字符，仍需检测
    if len(message.strip()) < 10:
        # 检查是否包含非拉丁字符，如果有则需要检测
        import re
        has_special_chars = bool(re.search(r'[^\x00-\x7F]', message))
        if not has_special_chars:
            return False
    
    # 如果是纯数字或符号，跳过检测
    if message.strip().isdigit() or not any(c.isalpha() for c in message):
        return False
    
    # 如果当前语言为空或默认值，需要检测
    if not current_language or current_language in ['English']:
        return True
    
    # 增强的语言特征检测，支持更多小语种
    if current_language and current_language != 'English':
        import re
        
        # 扩展的语言特征检测
        language_patterns = {
            'chinese': r'[\u4e00-\u9fff]',
            'japanese': r'[\u3040-\u309f\u30a0-\u30ff]',
            'korean': r'[\uac00-\ud7af]',
            'arabic': r'[\u0600-\u06ff]',
            'thai': r'[\u0e00-\u0e7f]',
            'vietnamese': r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
            'hindi': r'[\u0900-\u097f]',
            'bengali': r'[\u0980-\u09ff]',
            'tamil': r'[\u0b80-\u0bff]',
            'telugu': r'[\u0c00-\u0c7f]',
            'malayalam': r'[\u0d00-\u0d7f]',
            'kannada': r'[\u0c80-\u0cff]',
            'gujarati': r'[\u0a80-\u0aff]',
            'punjabi': r'[\u0a00-\u0a7f]',
            'urdu': r'[\u0600-\u06ff\u0750-\u077f]',
            'persian': r'[\u0600-\u06ff\u0750-\u077f]',
            'russian': r'[а-яё]',
            'greek': r'[\u0370-\u03ff]',
            'hebrew': r'[\u0590-\u05ff]',
            'myanmar': r'[\u1000-\u109f]',
            'khmer': r'[\u1780-\u17ff]',
            'lao': r'[\u0e80-\u0eff]',
            'georgian': r'[\u10a0-\u10ff]',
            'armenian': r'[\u0530-\u058f]',
            'ethiopic': r'[\u1200-\u137f]',
        }
        
        # 检测当前消息中的语言特征
        detected_features = []
        for lang, pattern in language_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                detected_features.append(lang)
        
        # 如果检测到与当前语言不匹配的特征，需要重新检测
        current_lang_mapping = {
            'zh': 'chinese',
            'ja': 'japanese', 
            'ko': 'korean',
            'ar': 'arabic',
            'th': 'thai',
            'vi': 'vietnamese',
            'hi': 'hindi',
            'bn': 'bengali',
            'ta': 'tamil',
            'te': 'telugu',
            'ml': 'malayalam',
            'kn': 'kannada',
            'gu': 'gujarati',
            'pa': 'punjabi',
            'ur': 'urdu',
            'fa': 'persian',
            'ru': 'russian',
            'el': 'greek',
            'he': 'hebrew',
            'my': 'myanmar',
            'km': 'khmer',
            'lo': 'lao',
            'ka': 'georgian',
            'hy': 'armenian',
            'am': 'ethiopic',
        }
        
        current_feature = current_lang_mapping.get(current_language)
        if detected_features and current_feature not in detected_features:
            return True
    
    # 其他情况下，如果已有语言设置，跳过检测以提升性能
    return False

async def detect_and_update_language(user_message: str, current_user_data: dict, user_ip: str = "") -> dict:
    """
    使用专门的AI检测用户输入的语言并更新language字段（支持智能缓存）
    """
    current_language = current_user_data.get("language", "English")

    # 检查是否需要进行语言检测
    if not should_detect_language(user_ip, user_message, current_language):
        logger.info(f"跳过语言检测: 消息过短或不需要检测")
        return current_user_data
    
    # 增强的语言缓存检查
    if user_ip:
        cache_key = f"lang_{user_ip}"
        # 检查专用语言缓存
        if cache_key in _language_cache:
            cache_entry = _language_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 86400:  # 24小时缓存
                cached_language = cache_entry['language']
                if cached_language and cached_language != "English":
                    if cached_language != current_language:
                        logger.info(f"⚡ 使用语言缓存: {current_language} -> {cached_language}")
                        current_user_data["language"] = cached_language
                    return current_user_data
            else:
                del _language_cache[cache_key]
        
        # 检查旧版全局缓存（兼容性）
        if cache_key in _global_cache:
            cache_entry = _global_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 86400:  # 24小时缓存
                cached_language = cache_entry['language']
                if cached_language and cached_language != "English":
                    if cached_language != current_language:
                        logger.info(f"使用缓存的语言设置: {current_language} -> {cached_language}")
                        current_user_data["language"] = cached_language
                    return current_user_data
            else:
                del _global_cache[cache_key]

    # 创建增强的语言检测提示词，支持更多小语种
    language_detection_prompt = f"""你是一个专业的多语言检测助手。请分析用户输入文本，识别其主要使用的语言。

用户输入："{user_message}"
当前language字段：{current_language}

检测规则：
1. 识别用户输入的主要语言（包括各种大语种和小语种）
2. 使用标准的语言代码格式：
   - 中文（简体/繁体）→ "zh"
   - 英文 → "English"  
   - 日文 → "ja"
   - 韩文 → "ko"
   - 阿拉伯文 → "ar"
   - 泰文 → "th"
   - 越南文 → "vi"
   - 印地文 → "hi"
   - 孟加拉文 → "bn"
   - 泰米尔文 → "ta"
   - 泰卢固文 → "te"
   - 马拉雅拉姆文 → "ml"
   - 卡纳达文 → "kn"
   - 古吉拉特文 → "gu"
   - 旁遮普文 → "pa"
   - 乌尔都文 → "ur"
   - 波斯文 → "fa"
   - 俄文 → "ru"
   - 希腊文 → "el"
   - 希伯来文 → "he"
   - 缅甸文 → "my"
   - 高棉文 → "km"
   - 老挝文 → "lo"
   - 格鲁吉亚文 → "ka"
   - 亚美尼亚文 → "hy"
   - 阿姆哈拉文 → "am"
   - 其他欧洲语言：es、fr、de、it、pt、nl、sv、no、da、fi、pl、tr、cs、sk、sl、hr、bg、mk、sr、bs、is、ga、cy、eu、ca、gl、mt、lv、lt、et、sq、af等

3. 特殊处理：
   - 混合语言文本：选择占主导地位的语言
   - 无法确定或纯符号/数字：保持当前language设置
   - 文本过短但有明确特征：优先检测结果
   - 小语种优先：如果检测到小语种特征，优先识别为小语种

4. 小语种识别要点：
   - 仔细识别南亚语言（印地文、孟加拉文、泰米尔文等）
   - 准确区分东南亚语言（泰文、越南文、缅甸文、高棉文等）
   - 正确识别中东语言（阿拉伯文、波斯文、乌尔都文、希伯来文等）
   - 精确区分东亚语言（中文、日文、韩文）

输出要求：
- 仅输出语言代码，不要解释
- 示例：zh 或 English 或 es 或 ja 或 th 或 hi"""

    try:
        # 调用语言检测AI - 使用HumanMessage而不是SystemMessage
        detection_response = await llm_for_language_detection.ainvoke([
            HumanMessage(content=language_detection_prompt)
        ])

        detected_language = detection_response.content.strip()
        logger.info(f"语言检测AI结果: '{detected_language}' (原语言: {current_language})")

        # 验证检测结果是否有效（简单验证：非空且合理长度）
        if detected_language and len(detected_language) <= 10 and detected_language.replace("-", "").replace("_",
                                                                                                             "").isalnum():
            # 缓存检测结果到专用语言缓存
            if user_ip and detected_language != "English":  # 只缓存非默认语言
                cache_key = f"lang_{user_ip}"
                _language_cache[cache_key] = {
                    'language': detected_language,
                    'timestamp': time.time()
                }
                # 限制语言缓存大小
                if len(_language_cache) > 1000:
                    oldest_key = min(_language_cache.keys(), 
                                   key=lambda k: _language_cache[k]['timestamp'])
                    del _language_cache[oldest_key]
            
            # 如果检测到的语言与当前不同，更新数据
            if detected_language != current_language:
                logger.info(f"语言更新: {current_language} -> {detected_language}")
                current_user_data["language"] = detected_language
                return current_user_data

        # 如果检测结果无效或相同，保持原状
        logger.info(f"语言保持不变: {current_language}")
        return current_user_data

    except Exception as e:
        logger.error(f"语言检测AI调用失败: {e}")
        # 出错时保持原有语言设置
        return current_user_data


async def send_customer_data_to_api(user_ip: str, user_data_dict: dict):
    """发送用户数据到客户信息创建API"""
    try:
        # 映射用户数据到API要求的格式，确保所有值都是字符串类型且处理None值
        def safe_get_str(value, default="0"):
            """安全获取字符串值，处理None情况"""
            if value is None or value == "":
                return default
            return str(value)
        
        api_data = {
            "ipAddress": user_ip,
            "name": safe_get_str(user_data_dict.get("name"), ""),
            "age": safe_get_str(user_data_dict.get("age"), "0"),
            "phone": safe_get_str(user_data_dict.get("phone"), ""),
            "email": safe_get_str(user_data_dict.get("email"), ""),
            "country": safe_get_str(user_data_dict.get("country"), ""),
            "language": safe_get_str(user_data_dict.get("language"), "zh"),
            "firstMedicalOpinion": safe_get_str(user_data_dict.get("firstMedicalOpinion"), "0"),
            "medicalAttachments": safe_get_str(user_data_dict.get("medicalAttachments"), "0"),
            "preferredCity": safe_get_str(user_data_dict.get("preferredCity"), "0"),
            "preferredHospital": safe_get_str(user_data_dict.get("preferredHospital"), "0"),
            "treatmentBudget": safe_get_str(user_data_dict.get("treatmentBudget"), "0"),
            "treatmentPreference": safe_get_str(user_data_dict.get("treatmentPreference"), "0"),
            "urgencyLevel": safe_get_str(user_data_dict.get("urgencyLevel"), "0"),
            "consultationType": safe_get_str(user_data_dict.get("consultationType"), "0"),
            "remark": safe_get_str(user_data_dict.get("remark"), "0"),
        }

        logger.info(f"准备发送用户数据到API: {api_data}")

        # 发送HTTP POST请求
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://localhost:48080/admin-api/datamanagement/customer-info/create", #传入数据地址
                json=api_data,
                headers={
                    "Content-Type": "application/json",
                    "tenant-id": "1"
                }
            )

            if response.status_code == 200:
                logger.info(f"成功发送用户数据到API: 用户IP={user_ip}, 响应={response.text}")
            else:
                logger.error(f"发送用户数据到API失败: 状态码={response.status_code}, 响应={response.text}")

    except Exception as e:
        logger.error(f"发送用户数据到API时发生异常: {e}")


def clean_tool_messages_from_record(record: list) -> list:
    """
    清理对话记录中的工具相关消息，避免DashScope报错：
    "An assistant message with tool_calls must be followed by tool messages"
    
    - 删除所有 ToolMessage
    - 同时删除包含 tool_calls 的 AIMessage（防止出现“缺少对应工具响应”的历史消息）
    
    Args:
        record: 包含各种消息类型的对话记录列表
        
    Returns:
        list: 删除工具消息和相关assistant工具调用后的对话记录
    """
    cleaned_record = []
    for msg in record:
        # 跳过工具消息
        if isinstance(msg, ToolMessage):
            continue

        # 删除包含tool_calls的AI消息，避免历史中出现未配对的工具调用
        try:
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage):
                has_native_tool_calls = bool(getattr(msg, "tool_calls", None))
                has_kw_tool_calls = bool(getattr(msg, "additional_kwargs", {}) and msg.additional_kwargs.get("tool_calls"))
                if has_native_tool_calls or has_kw_tool_calls:
                    continue
        except Exception:
            pass

        cleaned_record.append(msg)

    logger.info(f"清理工具消息: 原始记录{len(record)}条，清理后{len(cleaned_record)}条")
    return cleaned_record


def cleanup_database_record(database: dict, user_ip: str) -> None:
    """
    清理数据库中指定用户的工具记录
    
    Args:
        database: 用户数据库
        user_ip: 用户IP地址
    """
    if user_ip in database and "record" in database[user_ip]:
        original_count = len(database[user_ip]["record"])
        database[user_ip]["record"] = clean_tool_messages_from_record(database[user_ip]["record"])
        cleaned_count = len(database[user_ip]["record"])
        logger.info(f"用户 {user_ip} 的记录已清理: {original_count} -> {cleaned_count}")

