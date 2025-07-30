import httpx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import logging
from mcp import ClientSession

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent-Server")


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
        template="""你是一个专业的医疗科普助手。

                ## 🚨🚨🚨 强制工具调用指令 🚨🚨🚨
                
                对于用户查询："{query}"
                
                如果用户问题涉及任何医疗内容（疾病、症状、治疗、健康建议等），你必须：
                1. 首先调用 medical_qa_search 工具
                2. 使用工具返回的信息来回答用户问题
                3. 绝对不允许不调用工具就直接回答医疗问题
                
                **当前查询包含医疗关键词，立即调用工具！**
                
                ## 当前用户数据状态
                {data}
                
                ## 历史对话记录
                {record}

                ## 用户最新消息
                {query}
                
                ## 用户偏好语言：{language}
                - 如果language="zh" → 用中文回复
                - 如果language="English" → 用英文回复

                **记住：先调用medical_qa_search工具，再回复！**\n""",
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

def clean_conversation_history(conversation_history: list) -> str:
    """彻底清理对话历史，只保留核心对话内容"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    
    history_text = ""
    
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            # 如果content为空但有工具调用，跳过这条消息
            if not msg.content.strip():
                continue
            # 提取AI回复的核心内容
            clean_reply = extract_reply_from_ai_message(msg.content)
            if clean_reply.strip():  # 只有非空回复才添加
                history_text += f"助手: {clean_reply}\n"
        elif isinstance(msg, ToolMessage):
            # 跳过工具消息，因为它们是技术细节
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
    """生成预制词的提示模板"""
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
                 4. 必须包括一个礼貌的拒绝选项（用对应语言的标准表达，如中文"不愿透露"、英文"Prefer not to say"等）
                 5. 预制词要简短明了，便于用户快速选择
                 6. 智能生成选项：
                    - 选择题：提供标准选项（用对应语言）
                    - 个人信息：提供"输入"和"跳过"选项  
                    - 年龄：提供年龄段选项
                    - 地理位置：提供常见选项
                    - 预算：提供合理范围选项
                 7. **语言一致性**：所有预制词必须使用传入的language对应的语言进行输出，不得混用
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
def get_follow_up_prompt():
    """生成后续对话的提示模板"""
    follow_up_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的医疗科普助手。请根据工具返回的结果和用户问题和用户数据，提供准确详细的定制化医疗回复。


            ## 回复语言规范：
            - **严格按照language字段回复**：请查看用户数据中的language字段，严格使用该语言进行回复"""),
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
                            tool_call_id=tool_call["id"],
                            name=func_name,  # 添加name字段，千帆AI要求function消息必须有name
                        )
                    )
                    tool_found = True
                    break

            if not tool_found:
                logger.warning(f"工具未找到: {func_name}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {func_name} not found",
                        tool_call_id=tool_call["id"],
                        name=func_name,  # 添加name字段，千帆AI要求function消息必须有name
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
                    name=func_name,  # 添加name字段，千帆AI要求function消息必须有name
                )
            )
    return tool_messages

# 使用专门的AI检测用户输入的语言并更新language字段
llm_for_language_detection = QianfanChatEndpoint(model="ERNIE-4.5-Turbo-128K", temperature=0.1)
async def detect_and_update_language(user_message: str, current_user_data: dict) -> dict:
    """
    使用专门的AI检测用户输入的语言并更新language字段
    """
    current_language = current_user_data.get("language", "English")

    # 创建语言检测提示词
    language_detection_prompt = f"""你是一个专业的多语言检测助手。请分析用户输入文本，识别其主要使用的语言。

用户输入："{user_message}"
当前language字段：{current_language}

检测规则：
1. 识别用户输入的主要语言（包括各种大语种和小语种）
2. 使用标准的语言代码格式：
   - 中文（简体/繁体）→ "zh"
   - 英文 → "English"  
   - 其他语言使用ISO 639-1代码（如：es、fr、de、ja、ko、it、pt、ru、ar、hi、th、vi、nl、sv、no、
   da、fi、pl、tr、he、fa、ur、bn、ta、te、ml、kn、gu、mr、ne、si、my、km、lo、ka、am、sw、zu、
   xh、af、sq、bg、hr、cs、et、lv、lt、mt、sk、sl、mk、sr、bs、is、ga、cy、eu、ca、gl等）

3. 特殊处理：
   - 混合语言文本：选择占主导地位的语言
   - 无法确定或纯符号/数字：保持当前language设置
   - 文本过短但有明确特征：优先检测结果

输出要求：
- 仅输出语言代码，不要解释
- 示例：zh 或 English 或 es 或 ja"""

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

