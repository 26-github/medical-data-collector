from langchain_mcp_adapters.client import MultiServerMCPClient
from fastapi import FastAPI
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools


from method import *
from api import *



# 初始化llm
llm_for_data = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_for_generating_preset = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_for_translate = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  # 专门用于语言检测的AI

# 拿到提示词模板
prompt_collecting = get_prompt_for_collecting_data()
prompt_prefabricated_words = get_prompt_for_generating_preset()



tools = None
mcp_server = None

async def tool_list():
    """初始化MCP客户端和工具"""
    global tools, mcp_server
    mcp_server = MultiServerMCPClient({
        "date": {
            "url": "http://localhost:8001/sse", # MCP服务器地址
            "transport": "sse",
        }
    })
    
    # 获取MCP工具
    async with mcp_server.session("date") as session:
        tools = await load_mcp_tools(session)
        logger.info(f"已加载 {len(tools)} 个工具")



@asynccontextmanager
async def lifespan(app):
    """
    应用生命周期管理器，在启动时加载模型和工具
    """
    logger.info("应用启动中...")
    await tool_list()
    logger.info("Agent初始化完成，服务器准备就绪。")
    yield
    # 在这里可以添加应用关闭时需要执行的代码
    logger.info("应用正在关闭...")

app = FastAPI(lifespan=lifespan)

#建立一个空字典存储数据于内存中
database = {}

@app.post("/chat", response_model=ChatResponse)    #post请求，接收用户消息和IP，返回AI回复和用户数据
async def chat_with_ai(request: ChatRequest):
    # 获取用户IP和消息
    user_ip = request.user_ip
    new_user_message = request.message
    input_language = request.language

    # 检查用户是否存在，如果不存在则创建新用户
    if check_user(user_ip, database)== False:
        database[user_ip]["data"]["language"] = input_language
    # 获取用户数据，如果不存在则初始化为空字典
    user_data_dict = database[user_ip].get("data", {})

    #如果用户对话超过十次，回复请找人工客服并发送数据到API
    if len(database[user_ip]["record"]) > 200000:
        language = user_data_dict.get("language", "zh")
        messages = [
            SystemMessage(f"将'对话次数过多，请寻找人工客服'这句话翻译成{language}语言输出,最后结果不用带双引号"),
        ]
        translation_output = await llm_for_translate.ainvoke(messages)

        # 发送用户数据到API
        await send_customer_data_to_api(user_ip, user_data_dict)

        return ChatResponse(
            user_ip=user_ip,
            ai_reply=translation_output.content,
            user_data=user_data_dict
        )

    # 如果用户数据为空，创建新的数据对象
    if not user_data_dict:
        user_data = data()
    else:
        user_data = data.model_validate(user_data_dict)

    # 添加用户消息到记录中
    user_message = HumanMessage(content=new_user_message)
    database[user_ip]["record"].append(user_message)

    # 正确绑定工具到LLM
    llm_with_tools = llm_for_data.bind_tools(tools)
    
    # 创建处理链
    chain = prompt_collecting | llm_with_tools
    
    # 准备输入数据，使用当前的用户数据
    input_data = {
        "query": new_user_message, 
        "data": user_data.model_dump(),
        "record": database[user_ip]["record"]
    }
    
    logger.info(f"开始并行调用语言检测AI和数据收集AI: 查询='{new_user_message}'")
    
    # **并行执行语言检测AI和数据收集AI**
    import asyncio
    language_task = asyncio.create_task(
        detect_and_update_language(new_user_message, user_data.model_dump())
    )
    data_collection_task = asyncio.create_task(
        chain.ainvoke(input_data)
    )
    
    # 等待两个AI调用完成
    updated_user_data_dict, output = await asyncio.gather(language_task, data_collection_task)
    
    # 智能合并结果：使用语言检测AI的language字段，保留数据收集可能的其他更新
    user_data_with_language = data.model_validate(updated_user_data_dict)
    logger.info(f"并行AI调用完成，语言检测结果: {user_data_with_language.language}")
    
    # 如果数据收集AI也有输出（在没有工具调用的情况下），预处理可能的数据更新
    # 这个会在后面的解析步骤中处理，这里我们主要确保language字段正确
    user_data = user_data_with_language
    logger.info(f"数据收集AI输出: {output}")


    # 处理工具调用 - 优先使用output.tool_calls，备用additional_kwargs
    tool_calls = getattr(output, 'tool_calls', None) or output.additional_kwargs.get("tool_calls", [])
    logger.info(f"检测到工具调用: {len(tool_calls)}个")
    if tool_calls:
        logger.info(f"工具调用详情: {tool_calls}")
    
    final_reply = output.content
    
    if tool_calls:
        # 使用正确的服务器创建会话，并重新加载工具
        async with mcp_server.session("date") as session:
            # 重新加载工具以确保session是活跃的
            fresh_tools = await load_mcp_tools(session)
            tool_responses = await execute_tools(tool_calls, session, fresh_tools)

        # 创建包含工具调用的消息列表
        messages_for_followup = database[user_ip]["record"] + [output] + tool_responses

        # 将提示词传入llm
        follow_up_prompt = get_follow_up_prompt()
        follow_up_chain = follow_up_prompt | llm_for_data

        # 再次调用模型
        follow_up_output = await follow_up_chain.ainvoke({"history": messages_for_followup})
        final_reply = follow_up_output.content

        # 更新记录
        database[user_ip]["record"].append(output)
        database[user_ip]["record"].extend(tool_responses)
        database[user_ip]["record"].append(follow_up_output)

    else:
        # 没有工具调用，直接添加到记录
        database[user_ip]["record"].append(output)
        logger.info("未检测到工具调用")

    # 解析数据更新
    formatted_output = parser_data.parse(final_reply)
    # 更新用户数据，但保护语言检测AI的language字段结果
    updated_data = formatted_output.model_dump(exclude_unset=True)
    
    # 保存语言检测AI的语言结果
    detected_language = user_data.language
    
    # 合并数据，但确保language字段优先使用语言检测AI的结果
    merged_data = {**user_data.model_dump(), **updated_data}
    merged_data["language"] = detected_language  # 强制使用语言检测AI的结果
    
    user_data = data.model_validate(merged_data)
    logger.info(f"最终数据合并完成，确保使用语言检测AI结果: {user_data.language}")

    # 更新数据库中的数据
    database[user_ip]["data"] = user_data.model_dump()

    # 使用解析出的回复
    final_reply = formatted_output.reply

    # 回复
    return ChatResponse(
        user_ip=user_ip,
        ai_reply=final_reply,
        user_data=user_data.model_dump(),
    )

@app.post("/preset_words", response_model=PresetWordsResponse)   #post请求，接收用户IP，返回预制词列表
async def get_preset_words(request: PresetWordsRequest):
    prompt_preset_words_and_model = prompt_prefabricated_words | llm_for_generating_preset
    # 获取用户ip并获得用户数据和消息
    user_ip = request.user_ip
    # 检查用户是否存在，如果不存在则创建新用户
    check_user(user_ip, database)
    data_and_message = database[user_ip]
    
    # 获取用户语言偏好用于日志记录
    user_language = data_and_message.get("data", {}).get("language", "English")
    logger.info(f"预制词生成请求 - 用户IP: {user_ip}, 语言: {user_language}")
    
    # 请求AI生成预制词（AI会自动根据language字段选择语言）
    preset_output = await prompt_preset_words_and_model.ainvoke({"data_and_ai_message": data_and_message})
    preset_words_output = parser_prefabricated_words.parse(preset_output.content)
    
    logger.info(f"预制词生成完成 - 生成数量: {len(preset_words_output.words)}")
    
    # 返回预制词
    return PresetWordsResponse(
        preset_words=preset_words_output.words
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8004, reload=True)
