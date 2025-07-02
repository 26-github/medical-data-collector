from pydantic import BaseModel, Field
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser



class data(BaseModel):
    """用户数据模型，包含要收集的用户信息和AI助手回复"""
    language: str = Field(default="English", description="用户的偏好语言")
    name: str = Field(default="0", description="用户的名字")
    gender: str = Field(default="0", description="用户的性别")
    age: str = Field(default="0", description="用户的年龄")
    phone_number: str = Field(default="0", description="用户的电话号码")
    disease: str = Field(default="0", description="用户的疾病")
    Severity_of_disease: str = Field(default="0", description="疾病的严重程度")
    budget: str = Field(default="0", description="用户用于治疗的预算")
    reply: str = Field(default="0", description="AI助手的回复")

class prefabricated_words(BaseModel):
    """根据ai的回复为用户生成三到五个预制词"""
    words: list[str] = Field(default_factory=list, description="预制词列表，内部包含三到五个预制词")

# 使用 PydanticOutputParser 将 data 模型解析为输出格式
parser_data = PydanticOutputParser(pydantic_object=data)
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

                ## 工作指引

                ### 1. 数据收集优先级（按顺序）：
                - 首先确认用户偏好语言 (language)(可以通过用户使用的语言自动识别)
                - 收集基本信息：姓名 (name)、年龄 (age)、性别 (gender)
                - 收集核心医疗信息：疾病 (disease)、病情严重程度 (Severity_of_disease)
                - 收集联系和预算信息：电话 (phone_number)、预算 (budget)

                ### 2. 对话策略：
                - 使用用户偏好的语言进行交流
                - 每次只询问一个问题，避免用户感到压力
                - 语调要友好、专业、体贴
                - 如果用户提供了多个信息，要全部更新
                - 先处理用户的回答，再提出下一个问题

                ### 3. 信息处理规则：
                - 如果用户拒绝回答或表示不愿意，记录为"unknown"
                - 如果用户提供的信息与之前矛盾，优先使用最新信息
                - 如果用户询问问题而不是回答，要先回答用户的问题再继续收集
                - 自动纠正明显的语言偏好（如用户用中文回答但language设为English）

                ### 4. 完成条件：
                - 当所有必要信息收集完毕后，提供一个友好的总结
                - 如果用户主动结束对话，要礼貌地确认并总结已收集的信息

                ### 5. 特殊情况处理：
                - 如果用户看起来困惑，要耐心解释收集信息的目的
                - 如果用户提供模糊信息，要礼貌地要求澄清
                - 对于敏感信息（如疾病），要表达理解和支持

                ## 输出要求：
                请根据以上指引更新用户数据，并在reply字段中写入你的回复。确保回复自然、友好且符合用户的语言偏好。

                {format_instructions}""",
        # query是用户最新消息，data是用户已收集的信息，record是对话记录
        input_variables=["query", "data", "record"],
        partial_variables={"format_instructions": parser_data.get_format_instructions()},
    )
    return prompt

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
                 2. 使用用户的偏好语言(language字段)生成预制词
                 3. 预制词应该是用户对AI问题的可能回答
                 4. 必须包括一个拒绝回答的选项（根据用户偏好语言：英语用"Prefer not to say"、中文用"不愿透露"等）
                 5. 预制词要简短明了，适合快速选择
                 6. 如果AI询问的是选择题，提供对应的选项
                 7. 如果AI在问姓名，提供常见姓名选项
                 8. 如果AI在问年龄，提供年龄段选项
                 9. 如果AI在问性别，提供性别选项
                 10. 严格要求：所有预制词都必须使用用户的偏好语言，绝对不能混用其他语言
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
def check_user(user_id,database):
    # 如果用户不存在，创建一个新的用户记录
    if user_id not in database:
        database[user_id] = {'record': [], 'data': {}}
