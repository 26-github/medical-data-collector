from langchain_mcp_adapters.client import MultiServerMCPClient
from fastapi import FastAPI
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools
import subprocess
import sys
import atexit
import asyncio
import httpx
from dotenv import load_dotenv
from typing import Any, Optional, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

from method import *
from api import *
from medical_attachment_processor import medical_processor
import re
import json
import hashlib
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta

# 导入专用缓存
from method import _response_cache


# 初始化llm（添加超时和重试配置）
# 注意：ERNIE模型可能不支持原生function calling，考虑使用支持的模型


# 如果需要function calling，考虑使用支持的模型：
# from langchain_openai import ChatOpenAI
# llm_for_data = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
llm_for_data = ChatOpenAI(
    model="qwen-max",
    temperature=0.1,
    request_timeout=60,
    max_retries=3,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
# 删除多余的LLM实例，只保留主要的数据收集LLM
# llm_for_generating_preset 和 llm_for_translate 已删除以提升性能

# 拿到提示词模板
prompt_collecting = get_prompt_for_collecting_data()
prompt_prefabricated_words = get_prompt_for_generating_preset()

# 简化的缓存管理器
class SimpleCache:
    def __init__(self):
        self.cache_dir = Path("simple_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.stats = {'hits': 0, 'misses': 0}
        
    def _generate_key(self, cache_type: str, identifier: str, extra: str = "") -> str:
        content = f"{cache_type}_{identifier}_{extra}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def get(self, cache_type: str, identifier: str, extra: str = "") -> Any:
        key = self._generate_key(cache_type, identifier, extra)
        
        # 检查内存缓存
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['created'] < entry['ttl']:
                self.stats['hits'] += 1
                logger.debug(f"缓存命中: {cache_type}")
                return entry['data']
            else:
                del self.memory_cache[key]
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if time.time() - entry['created'] < entry['ttl']:
                    self.memory_cache[key] = entry
                    self.stats['hits'] += 1
                    logger.debug(f"磁盘缓存命中: {cache_type}")
                    return entry['data']
                else:
                    cache_file.unlink()
            except:
                if cache_file.exists():
                    cache_file.unlink()
        
        self.stats['misses'] += 1
        return None
    
    async def set(self, cache_type: str, identifier: str, data: Any, ttl: int = 3600, extra: str = ""):
        key = self._generate_key(cache_type, identifier, extra)
        entry = {
            'data': data,
            'created': time.time(),
            'ttl': ttl
        }
        
        # 保存到内存
        self.memory_cache[key] = entry
        
        # 保存到磁盘
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"保存缓存到磁盘失败: {e}")
    
    def get_stats(self):
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': round(hit_rate, 2),
            'memory_entries': len(self.memory_cache)
        }

# 全局缓存实例
simple_cache = SimpleCache()



tools = None
mcp_server = None
server_process = None

def extract_tool_calls_from_text(content, available_tools):
    """
    从AI文本回复中解析工具调用指令
    适用于不支持原生function calling的模型
    """
    tool_calls = []
    
    # 定义可能的工具调用模式
    patterns = [
        r'调用工具：(\w+)\s*(?:参数：(.+?))?',
        r'使用(\w+)工具\s*(?:参数：(.+?))?',
        r'执行(\w+)\s*(?:参数：(.+?))?',
        r'需要调用(\w+)\s*(?:参数：(.+?))?',
    ]
    
    # 检查是否包含医疗相关关键词，如果是则强制调用medical_qa_search
    medical_keywords = ['医疗', '病情', '诊断', '治疗', '症状', '心脏', '超声', '报告', '检查']
    if any(keyword in content for keyword in medical_keywords):
        # 创建医疗查询工具调用
        tool_call = {
            "id": f"medical_qa_search_{len(tool_calls)}",
            "function": {
                "name": "medical_qa_search",
                "arguments": json.dumps({
                    "question": content[:200]  # 使用前200字符作为问题
                }, ensure_ascii=False)
            }
        }
        tool_calls.append(tool_call)
        logger.info("检测到医疗相关内容，自动触发medical_qa_search工具调用")
    
    # 尝试匹配明确的工具调用指令
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            tool_name = match[0] if isinstance(match, tuple) else match
            arguments = match[1] if isinstance(match, tuple) and len(match) > 1 else "{}"
            
            # 验证工具是否存在
            tool_names = [tool.name for tool in available_tools] if available_tools else []
            if tool_name in tool_names:
                try:
                    args_dict = json.loads(arguments) if arguments else {}
                except:
                    args_dict = {"query": arguments} if arguments else {}
                
                tool_call = {
                    "id": f"{tool_name}_{len(tool_calls)}",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args_dict, ensure_ascii=False)
                    }
                }
                tool_calls.append(tool_call)
    
    return tool_calls

async def check_mcp_server_health():
    """检查MCP服务器是否健康运行"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # FastMCP使用SSE端点，直接测试SSE连接
            async with httpx.AsyncClient(timeout=10.0) as client:  # 增加超时时间
                response = await client.get("http://127.0.0.1:18002/sse/")
                # 对于SSE端点，200或其他非4xx错误都表示服务器在运行
                if response.status_code < 400 or response.status_code == 502:
                    return True
        except Exception as e:
            logger.warning(f"MCP健康检查尝试 {attempt + 1}/{max_retries} 失败: {e}")
            # 如果SSE失败，尝试根路径测试
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("http://127.0.0.1:18002/")
                    if response.status_code < 500:  # 任何非服务器错误都表示服务器在运行
                        return True
            except Exception as e2:
                logger.warning(f"MCP根路径检查也失败: {e2}")
                
            # 如果不是最后一次尝试，等待一段时间再重试
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    return False

async def tool_list():
    """初始化MCP客户端和工具"""
    global tools, mcp_server
    
    # 添加连接重试机制
    max_retries = 10
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # 使用127.0.0.1而不是localhost，避免DNS解析问题
            mcp_server = MultiServerMCPClient({
                "medical_tools": {
                    "url": "http://127.0.0.1:18002/sse/", # 注意末尾的斜杠
                    "transport": "sse",
                }
            })

            # 获取MCP工具，添加超时控制
            async with mcp_server.session("medical_tools") as session:
                # 使用asyncio.wait_for添加超时控制
                tools = await asyncio.wait_for(
                    load_mcp_tools(session),
                    timeout=30.0  # 30秒超时
                )
                logger.info(f"已加载 {len(tools)} 个工具")
                return  # 成功连接，退出重试循环
                
        except asyncio.TimeoutError:
            logger.warning(f"MCP工具加载超时，尝试 {attempt + 1}/{max_retries}")
            # 清理可能存在的连接
            try:
                if mcp_server:
                    await mcp_server.close()
            except:
                pass
            mcp_server = None
            
        except Exception as e:
            logger.warning(f"MCP连接尝试 {attempt + 1}/{max_retries} 失败: {e}")
            # 清理可能存在的连接
            try:
                if mcp_server:
                    await mcp_server.close()
            except:
                pass
            mcp_server = None
            
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries - 1:
            logger.info(f"等待 {retry_delay} 秒后重试...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 10)  # 指数退避，最大10秒
        else:
            logger.error("所有MCP连接尝试都失败了，将在没有工具的情况下继续运行")
            tools = []  # 设置为空列表，避免后续错误



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
        ], cwd=os.path.dirname(server_path), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)

        logger.info(f"MCP服务器已启动，进程ID: {server_process.pid}")

        # 增加等待时间，让服务器完全启动
        time.sleep(8)

        # 检查进程是否还在运行
        if server_process.poll() is None:
            logger.info("MCP服务器进程运行正常")
            return True
        else:
            # 进程已经退出，获取错误信息
            stdout, stderr = server_process.communicate()
            logger.error(f"MCP服务器启动失败，退出码: {server_process.returncode}")
            if stdout:
                logger.error(f"标准输出: {stdout.decode()}")
            if stderr:
                logger.error(f"标准错误: {stderr.decode()}")
            return False

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
    
    # 测试S3连接
    logger.info("测试S3连接...")
    s3_connected = await medical_processor.test_s3_connection()
    if s3_connected:
        logger.info("✅ S3连接正常")
    else:
        logger.warning("⚠️ S3连接失败，医疗附件功能可能无法正常工作")
    
    # 启动MCP服务器
    if not start_mcp_server():
        logger.error("MCP服务器启动失败，但继续启动客户端...")
    else:
        # 等待并检查服务器健康状态
        for i in range(5):
            await asyncio.sleep(2)
            if await check_mcp_server_health():
                logger.info("✅ MCP服务器健康检查通过")
                break
            logger.info(f"等待MCP服务器就绪... ({i+1}/5)")
        else:
            logger.warning("⚠️ MCP服务器健康检查失败，但继续启动")
    
    # 注册退出时关闭服务器的函数
    atexit.register(stop_mcp_server)
    
    # 初始化MCP客户端和工具
    await tool_list()
    
    if tools and len(tools) > 0:
        logger.info(f"✅ Agent初始化完成，服务器准备就绪。已加载 {len(tools)} 个工具")
    else:
        logger.warning("⚠️ Agent初始化完成，但没有可用的工具。客户端将在基础模式下运行")
    
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
    medical_file = request.file  # 新增：获取医疗附件文件名

    # 检查用户是否存在，如果不存在则创建新用户
    user_exists = check_user(user_ip, database)
    
    # 获取用户数据字典
    if user_exists:
        # 用户存在，从数据库获取现有数据
        user_data_dict = database[user_ip]["data"]
    else:
        # 新用户，使用请求中的用户数据
        user_data_dict = user_data

    #如果用户对话超过二十次，回复请找人工客服并发送数据到API（增强多语言支持）
    if len(database[user_ip]["record"]) > 80:
        language = user_data_dict.get("language", "zh")
        # 扩展的多语言消息，支持更多小语种
        service_messages = {
            "zh": "对话次数过多，请寻找人工客服。",
            "English": "Too many conversations, please contact human customer service.",
            "ja": "会話回数が多すぎます。人間のカスタマーサービスにお問い合わせください。",
            "ko": "대화 횟수가 너무 많습니다. 인간 고객 서비스에 문의하십시오.",
            "ar": "عدد المحادثات كثير جداً، يرجى الاتصال بخدمة العملاء البشرية.",
            "th": "การสนทนามากเกินไป กรุณาติดต่อฝ่ายบริการลูกค้าที่เป็นมนุษย์",
            "vi": "Quá nhiều cuộc trò chuyện, vui lòng liên hệ dịch vụ khách hàng con người.",
            "hi": "बहुत अधिक बातचीत, कृपया मानव ग्राहक सेवा से संपर्क करें।",
            "bn": "অনেক বেশি কথোপকথন, দয়া করে মানব গ্রাহক সেবার সাথে যোগাযোগ করুন।",
            "ta": "அதிக உரையாடல்கள், மனித வாடிக்கையாளர் சேவையைத் தொடர்பு கொள்ளவும்.",
            "te": "చాలా ఎక్కువ సంభాషణలు, దయచేసి మానవ కస్టమర్ సేవను సంప్రదించండి.",
            "ml": "വളരെയധികം സംഭാഷണങ്ങൾ, ദയവായി മനുഷ്യ ഉപഭോക്തൃ സേവനത്തെ ബന്ധപ്പെടുക.",
            "kn": "ಹೆಚ್ಚು ಸಂಭಾಷಣೆಗಳು, ದಯವಿಟ್ಟು ಮಾನವ ಗ್ರಾಹಕ ಸೇವೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "gu": "ઘણી બધી વાતચીત, કૃપા કરીને માનવ ગ્રાહક સેવાનો સંપર્ક કરો.",
            "pa": "ਬਹੁਤ ਸਾਰੀਆਂ ਗੱਲਬਾਤ, ਕਿਰਪਾ ਕਰਕੇ ਮਨੁੱਖੀ ਗਾਹਕ ਸੇਵਾ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
            "ur": "بہت زیادہ گفتگو، براہ کرم انسانی کسٹمر سروس سے رابطہ کریں۔",
            "fa": "مکالمات زیادی، لطفاً با خدمات مشتریان انسانی تماس بگیرید.",
            "ru": "Слишком много разговоров, пожалуйста, обратитесь к человеческой службе поддержки.",
            "es": "Demasiadas conversaciones, por favor contacte al servicio al cliente humano.",
            "fr": "Trop de conversations, veuillez contacter le service client humain.",
            "de": "Zu viele Gespräche, bitte wenden Sie sich an den menschlichen Kundendienst.",
            "my": "စကားပြောမှုများစွာ၊ လူသားဖောက်သည်ဝန်ဆောင်မှုကို ဆက်သွယ်ပါ။",
            "km": "ការសន្ទនាច្រើនពេក សូមទាក់ទងសេវាកម្មអតិថិជនមនុស្ស។",
            "lo": "ການສົນທະນາຫຼາຍເກີນໄປ, ກະລຸນາຕິດຕໍ່ບໍລິການລູກຄ້າມະນຸດ.",
            "he": "יותר מדי שיחות, אנא פנה לשירות לקוחות אנושי.",
            "ka": "ძალიან ბევრი საუბარი, გთხოვთ დაუკავშირდეთ ადამიანურ მომხმარებელთა სერვისს.",
            "hy": "Շատ շատ զրույցներ, խնդրում ենք կապվել մարդկային հաճախորդների ծառայության հետ:",
            "am": "በጣም ብዙ ንግግሮች፣ እባክዎ የሰው ደንበኛ አገልግሎትን ያነጋግሩ።"
        }
        
        return ChatResponse(
            user_ip=user_ip,
            ai_reply=service_messages.get(language, service_messages["zh"]),
        )

    # 如果用户数据为空，创建新的数据对象
    if not user_data_dict:
        user_data = data()
    else:
        user_data = data.model_validate(user_data_dict)

    # 处理S3医疗附件（如果有）
    medical_attachment_info = None
    if medical_file:
        # 首先检查用户数据中是否已有同一文件的医疗附件分析结果
        if (user_data.medicalAttachmentFilename == medical_file and 
            user_data.medicalAttachmentAnalysis != "0" and
            user_data.medicalAttachmentAnalysis != ""):
            # 使用已保存的医疗附件分析结果
            logger.info(f"🎯 直接使用用户数据中的医疗附件分析: {medical_file}")
            medical_attachment_info = {
                "status": "success_from_user_data",
                "extracted_content": user_data.medicalAttachmentAnalysis,
                "original_filename": medical_file,
                "file_type": "s3_medical_attachment"
            }
            # 更新摘要
            attachment_summary = medical_processor.get_attachment_summary(medical_attachment_info)
            user_data.medicalAttachments = attachment_summary
        else:
            # 需要重新处理医疗附件
            try:
                logger.info(f"开始处理S3医疗附件: {medical_file}")
                # 从S3下载并处理医疗附件
                medical_attachment_info = await medical_processor.process_medical_attachment_from_s3(
                    user_ip=user_ip, 
                    filename=medical_file
                )
                
                if medical_attachment_info["status"] in ["success", "success_from_cache"]:
                    # 将医疗附件信息更新到用户数据中
                    attachment_summary = medical_processor.get_attachment_summary(medical_attachment_info)
                    user_data.medicalAttachments = attachment_summary
                    # 保存完整的分析结果和文件名
                    user_data.medicalAttachmentAnalysis = medical_attachment_info["extracted_content"]
                    user_data.medicalAttachmentFilename = medical_file
                    
                    # 记录处理方式
                    if medical_attachment_info["status"] == "success_from_cache":
                        logger.info(f"⚡ 医疗附件使用缓存结果: {medical_file}")
                    else:
                        logger.info(f"💾 医疗附件处理成功并已缓存: {medical_file}")
                else:
                    logger.error(f"医疗附件处理失败: {medical_attachment_info.get('extracted_content', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"处理S3医疗附件时出错: {e}")
                user_data.medicalAttachments = f"医疗附件处理失败: {str(e)}"
    elif (user_data.medicalAttachmentFilename != "0" and 
          user_data.medicalAttachmentAnalysis != "0" and
          user_data.medicalAttachmentAnalysis != ""):
        # 没有新的医疗文件，但用户数据中有之前的医疗附件信息，继续使用
        logger.info(f"🔄 继续使用之前的医疗附件分析: {user_data.medicalAttachmentFilename}")
        medical_attachment_info = {
            "status": "success_from_previous_data",
            "extracted_content": user_data.medicalAttachmentAnalysis,
            "original_filename": user_data.medicalAttachmentFilename,
            "file_type": "s3_medical_attachment"
        }
        # 更新摘要
        attachment_summary = medical_processor.get_attachment_summary(medical_attachment_info)
        user_data.medicalAttachments = attachment_summary

    # 在添加新用户消息前，清理之前的工具消息以防止提示词过长
    cleanup_database_record(database, user_ip)
    
    # 添加用户消息到记录中
    user_message = HumanMessage(content=new_user_message)
    database[user_ip]["record"].append(user_message)

    # 正确绑定工具到LLM
    if tools and len(tools) > 0:
        llm_with_tools = llm_for_data.bind_tools(tools)
        logger.info(f"工具绑定完成，可用工具数量: {len(tools)}")
        tool_names = [tool.name for tool in tools]
        logger.info(f"可用工具列表: {tool_names}")
    else:
        llm_with_tools = llm_for_data
        logger.warning("没有可用的工具，使用不带工具的LLM")
    
    # 创建处理链
    chain = prompt_collecting | llm_with_tools
    
    # 准备输入数据，使用当前的用户数据（包含可能的医疗附件信息）
    input_data = {
        "query": new_user_message, 
        "data": user_data.model_dump(),
        "record": database[user_ip]["record"],
        "language": user_data.language
    }
    
    # 如果有医疗附件信息，添加到查询上下文中并准备传递给RAG工具
    if medical_attachment_info and medical_attachment_info["status"] in ["success", "success_from_cache", "success_from_user_data", "success_from_previous_data"]:
        enhanced_query = f"""用户问题: {new_user_message}

⚠️ 注意：用户提供了医疗附件，系统将自动使用图像分析结果增强RAG检索。

🚨 强制要求：必须调用medical_qa_search工具，使用以下参数：
- question: "{new_user_message}"  
- image_analysis: 完整的医疗图像分析结果

这样RAG系统将基于图像内容进行增强检索！"""
        input_data["query"] = enhanced_query
        # 将医疗附件信息存储以便后续传递给工具
        input_data["medical_attachment_info"] = medical_attachment_info
        logger.info("已将医疗附件分析结果整合到查询中，准备传递给RAG工具")
    
    # 已移除token消耗统计
    
    logger.info(f"开始并行调用语言检测AI和数据收集AI: 查询='{new_user_message}'")
    
    # 增强的响应缓存检查
    import hashlib
    cache_content = f"{user_ip}_{new_user_message}_{user_data.language}_{medical_file or ''}"
    cache_key = hashlib.md5(cache_content.encode('utf-8')).hexdigest()
    
    # 检查专用响应缓存
    if cache_key in _response_cache:
        cache_entry = _response_cache[cache_key]
        if time.time() - cache_entry['timestamp'] < 1800:  # 30分钟缓存
            logger.info(f"⚡ 使用响应缓存: {new_user_message[:50]}...")
            return ChatResponse(
                user_ip=user_ip,
                ai_reply=cache_entry['response'],
            )
        else:
            del _response_cache[cache_key]
    
    # 检查旧版简单缓存（兼容性）
    old_cache_key = f"{user_ip}_{new_user_message}_{user_data.language}"
    cached_response = await simple_cache.get("response", old_cache_key)
    
    if cached_response:
        logger.info(f"使用缓存的完整响应: {new_user_message[:50]}...")
        return ChatResponse(
            user_ip=user_ip,
            ai_reply=cached_response,
        )
    
    # 简化的语言检测
    try:
        # 更新用户语言设置
        updated_user_data_dict = await detect_and_update_language(new_user_message, user_data.model_dump(), user_ip)
        user_data = data.model_validate(updated_user_data_dict)
        
        # 执行数据收集AI
        output = await asyncio.wait_for(
            chain.ainvoke(input_data),
            timeout=30.0
        )
        
    except Exception as e:
        logger.error(f"AI调用出现错误: {e}")
        # 使用降级方案
        language = user_data.language or "zh"
        fallback_messages = {
            "zh": "抱歉，系统暂时繁忙，请稍后重试。",
            "English": "Sorry, the system is temporarily busy. Please try again later.",
            "ja": "申し訳ございませんが、システムが一時的に混雑しています。後でもう一度お試しください。",
            "ko": "죄송합니다. 시스템이 일시적으로 바쁩니다. 나중에 다시 시도해 주세요.",
        }
        return ChatResponse(
            user_ip=user_ip,
            ai_reply=fallback_messages.get(language, fallback_messages["zh"]),
        )
    
    logger.info(f"对话AI输出: {output}")


    # 处理工具调用 - 优先使用output.tool_calls，备用additional_kwargs
    tool_calls = getattr(output, 'tool_calls', None) or output.additional_kwargs.get("tool_calls", [])
    
    # 如果没有原生工具调用，尝试解析文本中的工具调用指令
    if not tool_calls and tools and len(tools) > 0:
        logger.info(f"未检测到原生工具调用，尝试从文本解析。可用工具: {[tool.name for tool in tools]}")
        logger.info(f"AI回复内容预览: {output.content[:200]}...")
        tool_calls = extract_tool_calls_from_text(output.content, tools)
        if tool_calls:
            logger.info(f"从文本中解析到工具调用: {len(tool_calls)}个")
        else:
            logger.warning("未能从文本中解析到任何工具调用")
    
    logger.info(f"检测到工具调用: {len(tool_calls)}个")
    if tool_calls:
        logger.info(f"工具调用详情: {tool_calls}")
    
    final_reply = output.content
    
    if tool_calls and tools and len(tools) > 0 and mcp_server:
        try:
            # 添加调试日志
            logger.info(f"工具调用前检查: medical_attachment_info存在={medical_attachment_info is not None}")
            if medical_attachment_info:
                logger.info(f"医疗附件状态: {medical_attachment_info.get('status', 'unknown')}")
                logger.info(f"extracted_content长度: {len(medical_attachment_info.get('extracted_content', ''))}")
            
            # 如果有医疗附件信息且调用了medical_qa_search工具，增强工具参数
            if medical_attachment_info and medical_attachment_info["status"] in ["success", "success_from_cache"]:
                enhanced_tool_calls = []
                for tool_call in tool_calls:
                    if ("function" in tool_call and 
                        tool_call["function"]["name"] == "medical_qa_search"):
                        # 解析现有参数
                        import json
                        args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                        # 添加图像分析参数
                        args["image_analysis"] = medical_attachment_info["extracted_content"]
                        # 更新工具调用
                        enhanced_tool_call = tool_call.copy()
                        enhanced_tool_call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
                        enhanced_tool_calls.append(enhanced_tool_call)
                        logger.info(f"✅ 成功增强medical_qa_search工具调用，添加图像分析参数（长度: {len(args['image_analysis'])}）")
                        logger.info(f"增强后的工具参数: {args}")
                    else:
                        enhanced_tool_calls.append(tool_call)
                tool_calls = enhanced_tool_calls
            else:
                logger.warning(f"⚠️ 未能增强工具调用 - medical_attachment_info状态: {medical_attachment_info.get('status', 'None') if medical_attachment_info else 'None'}")
            
            # 使用正确的服务器创建会话，并重新加载工具
            async with mcp_server.session("medical_tools") as session:
                # 重新加载工具以确保session是活跃的
                fresh_tools = await load_mcp_tools(session)
                tool_responses = await execute_tools(tool_calls, session, fresh_tools)

            # 创建包含工具调用的消息列表
            messages_for_followup = database[user_ip]["record"] + [output] + tool_responses

            # 生成后续对话提示词（无需token统计）
            follow_up_prompt = get_follow_up_prompt()
            follow_up_chain = follow_up_prompt | llm_for_data

            # 再次调用模型
            follow_up_output = await follow_up_chain.ainvoke({"history": messages_for_followup})
            final_reply = follow_up_output.content

            # 更新记录
            database[user_ip]["record"].append(output)
            database[user_ip]["record"].extend(tool_responses)
            database[user_ip]["record"].append(follow_up_output)
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            logger.warning("工具调用失败，使用原始AI回复")
            database[user_ip]["record"].append(output)
            final_reply = output.content
    else:
        # 没有工具调用或工具不可用，直接添加到记录
        database[user_ip]["record"].append(output)
        if tool_calls:
            logger.warning("检测到工具调用但工具不可用，忽略工具调用")
        else:
            logger.info("未检测到工具调用")

    # 直接使用AI的文本回复，不需要解析JSON结构
    # 更新数据库中的数据（保持用户数据不变，只更新语言）
    database[user_ip]["data"] = user_data.model_dump()

    # 缓存响应结果到专用缓存（同步执行，更快）
    try:
        _response_cache[cache_key] = {
            'response': final_reply,
            'timestamp': time.time(),
            'original_question': new_user_message
        }
        # 限制响应缓存大小
        if len(_response_cache) > 500:
            oldest_key = min(_response_cache.keys(), 
                           key=lambda k: _response_cache[k]['timestamp'])
            del _response_cache[oldest_key]
        logger.info(f"💾 响应已缓存: {new_user_message[:30]}...")
    except Exception as e:
        logger.warning(f"缓存响应失败: {e}")
    
    # 同时异步缓存到旧版系统（兼容性）
    asyncio.create_task(simple_cache.set(
        "response", 
        old_cache_key, 
        final_reply, 
        ttl=1800  # 30分钟缓存
    ))

    # final_reply 已经是AI的直接回复

    # 回复
    return ChatResponse(
        user_ip=user_ip,
        ai_reply=final_reply,
    )
@app.get("/cache/stats")
async def get_cache_stats():
    """获取所有缓存统计信息"""
    try:
        # 获取简化缓存统计
        simple_cache_stats = simple_cache.get_stats()
        
        # 获取医疗附件缓存统计
        medical_cache_stats = medical_processor.get_cache_statistics()
        
        # 获取新增的专用缓存统计
        from method import _language_cache, _response_cache
        language_cache_stats = {
            "total_entries": len(_language_cache),
            "cache_type": "language_detection"
        }
        
        response_cache_stats = {
            "total_entries": len(_response_cache),
            "cache_type": "response_cache"
        }
        
        return {
            "status": "success",
            "simple_cache": simple_cache_stats,
            "medical_attachment_cache": medical_cache_stats,
            "language_cache": language_cache_stats,
            "response_cache": response_cache_stats,
            "performance_summary": {
                "total_cache_entries": (
                    simple_cache_stats.get("memory_entries", 0) + 
                    len(_language_cache) + 
                    len(_response_cache)
                ),
                "estimated_speedup": "30-80% faster responses for cached queries"
            }
        }
    except Exception as e:
        logger.error(f"获取缓存统计信息失败: {e}")
        return {
            "status": "error",
            "message": f"获取缓存统计信息失败: {str(e)}"
        }

@app.post("/cache/clear")
async def clear_cache():
    """清理所有缓存"""
    try:
        # 清理简化缓存
        cleared_count = len(simple_cache.memory_cache)
        simple_cache.memory_cache.clear()
        
        # 清理磁盘缓存
        import shutil
        if simple_cache.cache_dir.exists():
            shutil.rmtree(simple_cache.cache_dir)
            simple_cache.cache_dir.mkdir(exist_ok=True)
        
        # 清理新增的专用缓存
        from .method import _language_cache, _response_cache
        language_cleared = len(_language_cache)
        response_cleared = len(_response_cache)
        _language_cache.clear()
        _response_cache.clear()
        
        # 重置统计
        simple_cache.stats = {'hits': 0, 'misses': 0}
        
        total_cleared = cleared_count + language_cleared + response_cleared
        
        return {
            "status": "success",
            "message": f"已清理所有缓存",
            "details": {
                "simple_cache": cleared_count,
                "language_cache": language_cleared,
                "response_cache": response_cleared,
                "total_cleared": total_cleared
            }
        }
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        return {
            "status": "error",
            "message": f"清理缓存失败: {str(e)}"
        }

@app.delete("/cache/user/{user_ip}")
async def clear_user_cache(user_ip: str):
    """清理特定用户的医疗附件缓存"""
    try:
        removed_count = medical_processor.clear_user_medical_cache(user_ip)
        return {
            "status": "success",
            "message": f"已清理用户 {user_ip} 的 {removed_count} 个缓存条目",
            "removed_count": removed_count
        }
    except Exception as e:
        logger.error(f"清理用户缓存失败: {e}")
        return {
            "status": "error", 
            "message": f"清理用户缓存失败: {str(e)}"
        }

@app.get("/cache/check/{user_ip}/{filename}")
async def check_attachment_cache(user_ip: str, filename: str):
    """检查特定医疗附件是否已缓存"""
    try:
        is_cached = medical_processor.is_attachment_cached(user_ip, filename)
        return {
            "status": "success",
            "user_ip": user_ip,
            "filename": filename,
            "is_cached": is_cached
        }
    except Exception as e:
        logger.error(f"检查缓存状态失败: {e}")
        return {
            "status": "error",
            "message": f"检查缓存状态失败: {str(e)}"
        }

"""
#弃用的预制词
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
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("client:app", host="localhost", port=8004, reload=True)
