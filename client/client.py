from langchain_mcp_adapters.client import MultiServerMCPClient
from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools
import subprocess
import sys
import os
import time
import signal
import atexit
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from method import *
from api import *



# 初始化llm
llm_for_data = QianfanChatEndpoint(model="ERNIE-Functions-8K", temperature=0.1)
llm_for_generating_preset = QianfanChatEndpoint(model="ERNIE-X1-32K-Preview", temperature=0.1)
llm_for_translate = QianfanChatEndpoint(model="ERNIE-X1-32K-Preview", temperature=0.1)
  # 专门用于语言检测的AI

# 拿到提示词模板
prompt_collecting = get_prompt_for_collecting_data()
prompt_prefabricated_words = get_prompt_for_generating_preset()



tools = None
mcp_server = None
server_process = None

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



def start_mcp_server():
    """启动MCP服务器"""
    global server_process
    try:
        # 获取server.py的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(current_dir, "..", "mcp_server", "server.py")
        server_path = os.path.normpath(server_path)

        logger.info(f"正在启动MCP服务器: {server_path}")

        # 启动server.py进程
        server_process = subprocess.Popen([
            sys.executable, server_path
        ], cwd=os.path.dirname(server_path))

        logger.info(f"MCP服务器已启动，进程ID: {server_process.pid}")

        # 等待服务器启动（给一些时间让服务器完全启动）
        time.sleep(3)

        return True
    except Exception as e:
        logger.error(f"启动MCP服务器失败: {e}")
        return False

def stop_mcp_server():
    """停止MCP服务器"""
    global server_process
    if server_process:
        try:
            logger.info("正在关闭MCP服务器...")
            server_process.terminate()

            # 等待进程正常结束
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果5秒后还没结束，强制杀死进程
                server_process.kill()
                server_process.wait()

            logger.info("MCP服务器已关闭")
            server_process = None
        except Exception as e:
            logger.error(f"关闭MCP服务器时出错: {e}")

@asynccontextmanager
async def lifespan(app):
    """
    应用生命周期管理器，在启动时加载模型和工具
    """
    logger.info("应用启动中...")
    
    # 启动MCP服务器
    if not start_mcp_server():
        logger.error("MCP服务器启动失败，但继续启动客户端...")
    
    # 注册退出时关闭服务器的函数
    atexit.register(stop_mcp_server)
    
    # 初始化MCP客户端和工具
    await tool_list()
    logger.info("Agent初始化完成，服务器准备就绪。")
    
    yield
    
    # 应用关闭时的清理工作
    logger.info("应用正在关闭...")
    stop_mcp_server()

app = FastAPI(lifespan=lifespan)

#建立一个空字典存储数据于内存中
database = {}

@app.post("/chat", response_model=ChatResponse)    #post请求，接收用户消息和IP，返回AI回复和用户数据
async def chat_with_ai(request: ChatRequest):
    # 获取用户IP和消息
    user_ip = request.user_ip
    new_user_message = request.message
    user_data=request.user_data

    # 检查用户是否存在，如果不存在则创建新用户
    user_exists = check_user(user_ip, database)
    
    # 获取用户数据字典
    if user_exists:
        # 用户存在，从数据库获取现有数据
        user_data_dict = database[user_ip]["data"]
    else:
        # 新用户，使用请求中的用户数据
        user_data_dict = user_data

    #如果用户对话超过二十次，回复请找人工客服并发送数据到API
    if len(database[user_ip]["record"]) > 80:
        language = user_data_dict.get("language", "zh")
        messages = [
            SystemMessage(f"将'对话次数过多，请寻找人工客服'这句话翻译成{language}语言输出,最后结果不用带双引号"),
        ]
        translation_output = await llm_for_translate.ainvoke(messages)


        return ChatResponse(
            user_ip=user_ip,
            ai_reply=translation_output.content,
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
    logger.info(f"工具绑定完成，可用工具数量: {len(tools)}")
    if tools:
        tool_names = [tool.name for tool in tools]
        logger.info(f"可用工具列表: {tool_names}")
    
    # 创建处理链
    chain = prompt_collecting | llm_with_tools
    
    # 准备输入数据，使用当前的用户数据
    input_data = {
        "query": new_user_message, 
        "data": user_data.model_dump(),
        "record": database[user_ip]["record"],
        "language": user_data.language
    }
    
    logger.info(f"开始并行调用语言检测AI和数据收集AI: 查询='{new_user_message}'")
    
    # **并行执行语言检测AI和对话AI**
    import asyncio
    language_task = asyncio.create_task(
        detect_and_update_language(new_user_message, user_data.model_dump())
    )
    data_collection_task = asyncio.create_task(
        chain.ainvoke(input_data)
    )
    
    # 等待两个AI调用完成
    updated_user_data_dict, output = await asyncio.gather(language_task, data_collection_task)
    
    # 智能合并结果：使用语言检测AI的language字段更新用户数据
    user_data_with_language = data.model_validate(updated_user_data_dict)
    logger.info(f"并行AI调用完成，语言检测结果: {user_data_with_language.language}")
    
    # 合并结果：保持语言检测AI的结果，其他数据保持不变
    user_data = user_data_with_language
    logger.info(f"对话AI输出: {output}")


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

    # 直接使用AI的文本回复，不需要解析JSON结构
    # 更新数据库中的数据（保持用户数据不变，只更新语言）
    database[user_ip]["data"] = user_data.model_dump()

    # final_reply 已经是AI的直接回复

    # 回复
    return ChatResponse(
        user_ip=user_ip,
        ai_reply=final_reply,
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
    uvicorn.run("client:app", host="localhost", port=8004, reload=True)
