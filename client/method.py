import httpx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
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
    reply: str = Field(default="0", description="AI助手的回复")

class prefabricated_words(BaseModel):           #生成预制词模型
    """根据ai的回复为用户生成三到五个预制词"""
    words: list[str] = Field(default_factory=list, description="预制词列表，内部包含三到五个预制词")

# 使用 PydanticOutputParser 将 data 模型解析为输出格式
parser_data = PydanticOutputParser(pydantic_object=data)
# 使用 PydanticOutputParser 将 prefabricated_words 模型解析为输出格式
parser_prefabricated_words = PydanticOutputParser(pydantic_object=prefabricated_words)

def get_prompt_for_collecting_data():  #生成收集数据的提示模板
    prompt = PromptTemplate(
        template="""你是一个专业的医疗咨询数据收集助手，目标是收集用户的健康相关信息以提供更好的建议。

                ## 当前用户数据状态
                {data}
                注意：字段值为"0"表示信息未收集，"unknown"表示用户拒绝提供

                ## 历史对话记录
                {record}

                ## 用户最新消息
                {query}

                ## 🎯 核心工作指引

                ### 1. 回复语言规范（最高优先级）：
                - **严格按照language字段回复**：系统已通过专门的语言检测AI设置了language字段，请严格使用该语言回复
                - **语言自动适配**：根据language字段的值，自动使用对应的语言进行回复：
                  * "zh" → 中文 
                  * "English" → 英文
                  * "es" → 西班牙语
                  * "fr" → 法语
                  * "ja" → 日语
                  * "ko" → 韩语
                  * 其他ISO语言代码 → 对应的语言
                - **通用语言质量要求**：
                  * 保持专业、友好、体贴的语调
                  * 使用该语言的标准表达方式
                  * 符合该语言文化的礼貌用语
                - **重要约束**：绝对不要修改language字段，该字段由专门的语言检测AI管理

                ### 2. 数据收集优先级和策略：
                **第一阶段（基础信息）**：
                - ✅ language（语言偏好）- 系统自动检测
                - 📝 name（姓名）- 建立信任关系
                - 📝 age（年龄）- 了解用户年龄段
                - 📝 country（国家）- 确定医疗资源范围

                **第二阶段（医疗需求）**：
                - 📝 firstMedicalOpinion（首次医疗意见）- 核心医疗需求
                - 📝 urgencyLevel（紧急程度）- 优先级评估
                - 📝 consultationType（咨询类型）- 服务方式偏好

                **第三阶段（就医偏好）**：
                - 📝 preferredCity（偏好就医城市）
                - 📝 preferredHospital（偏好医院）
                - 📝 treatmentPreference（治疗方案偏向）
                - 📝 treatmentBudget（治疗预算）

                **第四阶段（联系信息）**：
                - 📝 phone（电话）- 敏感信息，需谨慎询问
                - 📝 email（邮箱）- 后续联系方式

                ### 3. 智能对话策略：
                - **单一问题原则**：每次只问一个主要问题，避免用户感到负担
                - **自然过渡**：根据用户回答自然引导到下一个话题
                - **信息整合**：如果用户一次提供多个信息，全部更新到相应字段
                - **优先回应**：先处理用户的问题/关切，再继续数据收集
                - **智能判断**：从用户的话语中推断隐含信息（如从医院咨询推断就医偏好）

                ### 4. 特殊情况处理：
                - **拒绝回答**：记录为"unknown"，表达理解，转向其他话题
                - **信息冲突**：使用最新信息，礼貌确认变更
                - **模糊回答**：友好地请求澄清，提供具体选项
                - **敏感话题**：对医疗意见、预算等敏感信息表达理解和支持

                ### 5. 数据验证规则：
                对于有限选项的字段，确保回答符合规范：
                - **treatmentPreference**: "standard"(标准)/"advanced"(先进)/"clinical"(临床)
                - **urgencyLevel**: "low"(低)/"medium"(中)/"high"(高)/"emergency"(紧急)
                - **consultationType**: "online_only"(仅咨询)/"offline_required"(需要线下就医)

                ### 6. 工具使用策略：
                - **医院查询**：用户询问医院时调用get_hospital_info工具
                - **时间查询**：需要日期时间时调用相应工具
                - **策略整合**：使用工具结果回答问题 + 推断用户偏好 + 继续数据收集

                ### 7. 完成和总结：
                - **进度感知**：当收集到关键信息时，给予积极反馈
                - **适时总结**：重要信息收集完毕后，提供友好的确认总结
                - **服务导向**：始终关注如何更好地为用户提供医疗建议

                ## ⚠️ 重要输出要求 ⚠️
                1. **严格JSON格式**：必须输出有效的JSON，包含所有必要字段
                2. **语言一致性**：reply字段必须使用用户的偏好语言
                3. **信息完整性**：更新所有从用户消息中提取的信息
                4. **回复质量**：reply要自然、友好、专业，符合用户的文化背景

                {format_instructions}\n""",
        # query是用户最新消息，data是用户已收集的信息，record是对话记录
        input_variables=["query", "data", "record"],
        partial_variables={"format_instructions": parser_data.get_format_instructions()},
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
            ("system", """你是一个专业的医疗咨询数据收集助手。请根据工具返回的结果和用户问题，更新用户数据并提供回复。

            ## 核心任务提醒：
            你的主要职责是收集用户的医疗相关信息，工具调用只是为了更好地回答用户问题。在回答问题的同时，必须继续数据收集工作。

            ## 数据收集优先级（按顺序）：
            1. 首先确认用户偏好语言 (language) - **根据用户使用的语言自动识别和更新**
            2. 收集基本信息：姓名 (name)、年龄 (age)、国家 (country)
            3. 收集联系信息：电话 (phone)、邮箱 (email)  
            4. 收集医疗需求：第一医疗意见 (firstMedicalOpinion)、紧急程度 (urgencyLevel)
            5. 收集就医偏好：倾向就医的城市 (preferredCity)、倾向就医的医院 (preferredHospital)
            6. 收集治疗偏好：治疗预算 (treatmentBudget)、治疗方案倾向 (treatmentPreference)、咨询类型 (consultationType)

            ## 回复语言规范：
            - **严格按照language字段回复**：请查看用户数据中的language字段，严格使用该语言进行回复
            - **预算应填具体的数字，如100000，不要填10万**
            - **通用语言适配**：根据language字段的值自动选择对应语言，支持所有语种和小语种
            - **语言质量要求**：使用标准的语言表达，保持专业友好的语调，符合该语言的文化特色
            - **约束**：不要修改language字段，该字段由专门的语言检测AI管理

            ## 信息更新规则：
            - 从用户的问题和对话中提取任何可用的个人信息并更新相应字段
            - 如果用户问医院相关问题，可能暗示其就医偏好，相应更新preferredCity或preferredHospital
            - 如果用户表达紧急性，更新urgencyLevel字段
            - 如果用户提及预算相关内容，更新treatmentBudget字段

            ## 回复策略：
            1. 首先使用工具返回的结果回答用户的问题，
            特别注意：数据库中地址为中文，查询地址时要转换为中文进行查询，如“shanghai”查询时使用“上海”，“beijing”查询时使用“北京”
            2. 根据对话内容智能更新用户数据字段  
            3. 如果用户询问医院信息，在回答后可以询问是否有就医意向，以收集preferredCity/preferredHospital
            4. 在回答问题的同时，自然地引导收集下一个缺失的重要信息
            5. 保持友好、专业的语调，使用用户偏好的语言

            ## ⚠️ 重要输出格式要求 ⚠️：
            **你必须严格按照JSON格式输出，绝对不能输出纯文本！**
            
            请严格按照以下JSON格式输出，确保所有字段都包含在内：
            
            {format_instructions}
            
            **格式检查清单：**
            ✓ 必须以 {{ 开始，以 }} 结束
            ✓ 所有字符串值必须用双引号包围
            ✓ reply字段包含你对用户的回复内容
            ✓ 其他字段包含更新后的用户数据
            ✓ 不要在JSON外添加任何说明文字
            
            ## 示例处理：
            - 用户用中文问"上海有什么好医院？" → 更新language为"zh"，回答医院信息，询问是否考虑在上海就医
            - 用户问医院等级 → 回答后可询问对医院等级的偏好，收集treatmentPreference信息
            - 用户表达担心 → 可能暗示urgencyLevel，适当更新并继续收集其他信息

            **关键**：工具只是辅助手段，数据收集和表格填写是你的核心任务！
            **再次强调**：你的输出必须是有效的JSON格式，不能是纯文本！"""),
            MessagesPlaceholder(variable_name="history"),
        ])
    return follow_up_prompt.partial(
        format_instructions=parser_data.get_format_instructions()
    )

# 执行工具调用
async def execute_tools(tool_calls: list, session: ClientSession, tools: list) -> list:
    """执行工具调用并生成ToolMessage"""
    tool_messages = []
    for tool_call in tool_calls:
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
                func_name = tool_call.get("name", "")
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
                )
            )
    return tool_messages

# 使用专门的AI检测用户输入的语言并更新language字段
llm_for_language_detection = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
        # 调用语言检测AI
        detection_response = await llm_for_language_detection.ainvoke([
            SystemMessage(content=language_detection_prompt)
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

