from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from method import *
from api import *



# 初始化llm
llm_for_data = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_for_generating_preset = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 拿到提示词模板
prompt_collecting = get_prompt_for_collecting_data()
prompt_prefabricated_words = get_prompt_for_generating_preset()

#建立一个空字典存储数据于内存中
database = {}
@app.post("/chat", response_model=ChatResponse)    #post请求，接收用户消息和ID，返回AI回复和用户数据
async def chat_with_ai(request: ChatRequest):
    # 获取用户ID和消息
    user_id = request.user_id
    new_user_message = request.message
    #将prompt和模型结合
    prompt_collecting_and_model = prompt_collecting | llm_for_data
    # 检查用户是否存在，如果不存在则创建新用户
    check_user(user_id,database)
    # 获取用户数据，如果不存在则初始化为空字典
    user_data_dict = database[user_id].get("data", {})

    #如果用户对话超过十次，回复请找人工客服
    if len(database[user_id]["record"]) > 20:
        return ChatResponse(
            user_id=user_id,
            ai_reply="对话次数过多，请找人工客服",
            user_data=user_data_dict
        )

    # 如果用户数据为空，创建新的数据对象
    if not user_data_dict:
        user_data = data()
    else:
        user_data = data.model_validate(user_data_dict)


    # 拿到ai回复
    output = prompt_collecting_and_model.invoke({"query": new_user_message, "data": user_data.model_dump(),
                                      "record": database[user_id]["record"]})
    formatted_output = parser_data.parse(output.content)
    # 添加用户消息到记录中
    database[user_id]["record"].append(HumanMessage(content=new_user_message))

    # 更新用户数据
    updated_data = formatted_output.model_dump(exclude_unset=True)
    user_data = data.model_validate({**user_data.model_dump(), **updated_data})
    
    # 更新数据库中的数据
    database[user_id]["data"] = user_data.model_dump()
    # 添加AI回复到记录中
    database[user_id]["record"].append(AIMessage(content=formatted_output.reply))


    # 回复
    return ChatResponse(
        user_id=user_id,
        ai_reply=formatted_output.reply,
        user_data=user_data.model_dump(),
    )

@app.post("/preset_words", response_model=PresetWordsResponse)   #post请求，接收用户ID，返回预制词列表
async def get_preset_words(request: PresetWordsRequest):
    prompt_preset_words_and_model = prompt_prefabricated_words | llm_for_generating_preset
    # 获取用户id并获得用户数据和消息
    user_id = request.user_id
    # 检查用户是否存在，如果不存在则创建新用户
    check_user(user_id, database)
    data_and_message = database[user_id]
    # 请求llm生成预制词
    preset_output = prompt_preset_words_and_model.invoke({"data_and_ai_message": data_and_message})
    preset_words_output = parser_prefabricated_words.parse(preset_output.content)
    # 返回预制词
    return PresetWordsResponse(
        preset_words=preset_words_output.words
    )
