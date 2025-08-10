from fastmcp import FastMCP
import sys
import logging
from datetime import datetime
import requests
import json
import os
import time
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 添加 get_db_qa 相关的导入
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START
from langgraph.graph import StateGraph
# 已改用阿里云DashScope模型，移除init_chat_model导入

# 禁用Hugging Face符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 设置详细的日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("FastMCP-Server")

# 直连阿里云 DashScope Embeddings（OpenAI兼容端点在部分版本下与通用客户端字段不一致，这里手写轻量实现）
class DashScopeEmbeddings:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        request_timeout_seconds: int = 30,
        sleep_between_requests_seconds: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = request_timeout_seconds
        self.sleep_seconds = sleep_between_requests_seconds

    def embed_query(self, text: str):
        # 确保为字符串
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)
        payload = {"model": self.model, "input": text}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        raise ValueError(f"无效的Embedding响应: {data}")

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            try:
                vec = self.embed_query(t)
            except Exception:
                # 兜底，避免单条失败中断批量
                vec = [0.0] * 1024
            embeddings.append(vec)
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
        return embeddings

# 创建MCP服务器
mcp = FastMCP(
    name="medical_assistant_tools",
    instructions="""
    专业医疗助手工具服务器 - 提供全面的医疗信息查询和辅助功能

    🏥 核心功能模块：
    
    1. 【医学知识问答】- medical_qa_search
       - 基于专业医学知识库的智能问答系统
       - 支持疾病症状、诊断、治疗方法等全方位查询
       - 使用先进的检索增强生成技术，提供准确可靠的医学信息
       - 适用场景：疾病科普、治疗指南、医学术语解释等
    
    2. 【医院信息查询】- get_hospital_info
       - 全国医院数据库查询服务
       - 支持多维度搜索：医院名称、地址、等级、类型等
       - 提供详细医院信息：联系方式、规模、性质、简介等
       - 适用场景：就医推荐、医院对比、医疗资源查询等
    
    3. 【时间辅助工具】
       - get_current_date: 获取当前日期，格式为YYYY年MM月DD日
       - get_current_time: 获取当前时间，格式为HH:MM:SS
       - 适用场景：医疗记录时间戳、预约时间参考等

    🎯 使用指南：
    - 医学问答：直接输入医学相关问题，如"糖尿病的症状有哪些？"
    - 医院查询：可按医院名称、地区、等级等条件搜索
    - 时间查询：无需参数，直接获取当前日期时间
    
    ⚠️ 重要说明：
    - 所有医学信息仅供参考，不能替代专业医疗建议
    - 建议在专业医生指导下进行诊断和治疗
    - 医院信息来源于公开数据库，如有变动请以官方为准
    """,
)

# 初始化QA检索系统
try:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "sk-c258c59319a44549bbea71470bc00e62")
    if not dashscope_api_key:
        logger.warning("DASHSCOPE_API_KEY not found, QA retrieval system will not be available")
        qa_system_available = False
    else:
        # 初始化阿里云DashScope模型（对话）- 添加超时和重试配置
        llm = ChatOpenAI(
            model="qwen-max",
            temperature=0.1,
            request_timeout=60,  # 设置60秒超时
            max_retries=3,       # 最大重试3次
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        
        # 使用阿里云的text-embedding-v4模型（直连DashScope Embeddings端点，避免兼容层字段差异导致400）
        embeddings = DashScopeEmbeddings(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="text-embedding-v4",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            request_timeout_seconds=30,
            sleep_between_requests_seconds=0.0,
        )
        logger.info("✅ 成功使用阿里云的text-embedding-v4模型")
        
        vector_store = Chroma(
            collection_name="aliyun_text_embedding_v4_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )
        
        # 阿里云DashScope不支持reranker，使用简单的相似度排序
        reranker = None
        
        # 为HyDE（假设性文档嵌入）创建一个提示
        hyde_prompt = PromptTemplate(
            template="""你是一位专业的医学专家。基于用户的问题，请生成一个详细、准确的假设性答案文档，这个答案应该：

            1. 直接回答用户的问题
            2. 包含相关的医学术语和专业表述
            3. 结构化地组织信息（如症状、诊断、治疗等）
            4. 语言风格类似于医学指南、教科书或专业文献
            5. 内容详实，涵盖问题的各个方面

            请注意：这个答案将用于检索相似的医学文档，因此请尽可能使用标准的医学表述和术语。

            用户问题：{question}

            假设性答案：""",
            input_variables=["question"],
        )

        answer_prompt = PromptTemplate(
            template="""你是一个智能科普助手，通过从文档中提取信息来为用户全面科普知识，方方面面都要讲到。只能使用提供的文档内容来回答问题，不能使用其他信息。

        上下文：
        {context}

        问题：{question}

        答案：""",
            input_variables=["context", "question"]
        )

        class State(TypedDict):  # 定义状态类型
            question: str
            context: List[Document]
            reranked_context: List[Document]  # 添加重排序后的文档字段
            answer: str

        def retrieve(state: State):  # QA增强的文档检索
            question = state["question"]
            # 强制字符串化，避免 embeddings 端口收到非字符串
            if not isinstance(question, str):
                try:
                    import json as _json
                    question = _json.dumps(question, ensure_ascii=False)
                except Exception:
                    question = str(question)
            logger.info(f"🔍 开始QA增强检索，问题：{question}")

            # 阶段1：直接检索QA文档（使用问题本身）
            logger.info("📋 阶段1：检索QA问答对...")
            try:
                all_qa_docs = vector_store.similarity_search(question, k=20)
            except Exception as e:
                logger.error(f"阶段1 similarity_search 出错: {e}")
                raise
            qa_docs = [doc for doc in all_qa_docs if doc.metadata.get('content_type') in ['generated_qa', 'target_qa']][:8]
            logger.info(f"找到 {len(qa_docs)} 个QA文档")

            # 阶段2：直接检索原始文档（跳过HyDE以提升速度）
            logger.info("📄 阶段2：直接检索原始文档（跳过HyDE生成）...")
            
            # 直接使用原始问题检索原始文档，跳过HyDE假设性答案生成
            try:
                all_original_docs = vector_store.similarity_search(question, k=20)
            except Exception as e:
                logger.error(f"原始文档检索出错: {e}")
                raise
            original_docs = [doc for doc in all_original_docs if
                             doc.metadata.get('content_type') not in ['generated_qa', 'target_qa']][:8]
            logger.info(f"⚡ 快速检索到 {len(original_docs)} 个原始文档")

            # 合并结果，QA文档优先
            all_docs = qa_docs + original_docs

            # 去重（基于内容相似性）
            unique_docs = []
            seen_content = set()

            for doc in all_docs:
                # 使用内容的前100个字符作为去重依据
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_docs.append(doc)

            logger.info(f"🎯 检索完成，共获得 {len(unique_docs)} 个唯一文档（QA: {len(qa_docs)}, 原始: {len(original_docs)}）")

            # 返回最多15个文档用于重排序
            return {"context": unique_docs[:15]}

        def rerank(state: State):  # 简单的基于相似度的重排序（阿里云DashScope模型）
            query = state["question"]
            docs = state["context"]

            # 阿里云DashScope模型不支持reranker，使用简单的相似度排序
            try:
                # 优先选择QA文档，然后选择原始文档
                qa_docs = [doc for doc in docs if doc.metadata.get('content_type') in ['generated_qa', 'target_qa']]
                original_docs = [doc for doc in docs if doc.metadata.get('content_type') not in ['generated_qa', 'target_qa']]
                
                # 将QA文档排在前面，原始文档排在后面
                reranked_docs = qa_docs[:6] + original_docs[:4]  # 最多6个QA文档 + 4个原始文档
                
                # 如果不足10个，补充剩余文档
                remaining_docs = [doc for doc in docs if doc not in reranked_docs]
                while len(reranked_docs) < min(10, len(docs)) and remaining_docs:
                    reranked_docs.append(remaining_docs.pop(0))

                logger.info(f"简单重排序完成，从 {len(docs)} 个文档中选择了 {len(reranked_docs)} 个")

                # 分析重排序后的文档类型
                qa_count = len(
                    [doc for doc in reranked_docs[:10] if doc.metadata.get('content_type') in ['generated_qa', 'target_qa']])
                original_count = len(reranked_docs[:10]) - qa_count
                logger.info(f"📊 重排序结果：QA文档 {qa_count} 个，原始文档 {original_count} 个")

                return {"reranked_context": reranked_docs[:10]}  # 确保返回最多10个文档

            except Exception as e:
                logger.error(f"重排序过程中出现错误: {e}")
                logger.info("使用原始文档顺序作为备选方案")
                # 如果重排序失败，返回前10个原始文档
                return {"reranked_context": docs[:10]}

        def generate(state: State):  # QA增强的答案生成
            # 使用重排序后的文档，智能组织QA和原始文档内容
            max_doc_length = 1200  # 每个文档最大字符数
            max_total_length = 8000  # 总上下文最大字符数

            formatted_docs = []
            total_length = 0
            qa_count = 0
            original_count = 0

            for i, doc in enumerate(state["reranked_context"], 1):
                content_type = doc.metadata.get('content_type', 'unknown')

                # 格式化文档内容，区分QA和原始文档
                if content_type in ['generated_qa', 'target_qa']:
                    qa_count += 1
                    doc_header = f"【QA问答对 {qa_count}】"
                    if content_type == 'target_qa':
                        doc_header += "（专业知识库）"
                    elif content_type == 'generated_qa':
                        doc_header += f"（来源：{doc.metadata.get('source_file', '未知')}）"
                else:
                    original_count += 1
                    file_type = doc.metadata.get('file_type', 'unknown')
                    source_file = doc.metadata.get('source_file', '未知')
                    doc_header = f"【原始文档 {original_count}】（{file_type.upper()}：{source_file}）"

                # 截断文档内容
                doc_content = doc.page_content[:max_doc_length]
                formatted_content = f"{doc_header}\n{doc_content}"

                # 检查是否超过总长度限制
                content_with_separator = formatted_content + "\n\n"
                if total_length + len(content_with_separator) <= max_total_length:
                    formatted_docs.append(formatted_content)
                    total_length += len(content_with_separator)
                else:
                    # 如果会超过限制，添加截断的内容
                    remaining_space = max_total_length - total_length - len(doc_header) - 4  # 4 for "\n...\n\n"
                    if remaining_space > 100:
                        truncated_content = doc_content[:remaining_space] + "..."
                        formatted_content = f"{doc_header}\n{truncated_content}"
                        formatted_docs.append(formatted_content)
                    break

            docs_content = "\n\n".join(formatted_docs)
            logger.info(f"💡 生成答案，使用上下文：{len(docs_content)} 字符（QA文档: {qa_count}, 原始文档: {original_count}）")

            # 使用链式调用，避免将 PromptValue 直接传入模型导致的 contents 类型错误
            # 显式格式化为字符串再调用模型，避免将 PromptValue 直接传给模型
            formatted_answer_prompt = answer_prompt.format(
                question=state["question"],
                context=docs_content,
            )
            if not isinstance(formatted_answer_prompt, str):
                formatted_answer_prompt = str(formatted_answer_prompt)
            try:
                response = llm.invoke(formatted_answer_prompt)
            except Exception as e:
                logger.error(f"答案生成模型调用出错: {e}")
                raise
            return {"answer": response}

        # 构建状态图
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("rerank", rerank)
        graph_builder.add_node("generate", generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "rerank")
        graph_builder.add_edge("rerank", "generate")
        qa_graph = graph_builder.compile()
        
        qa_system_available = True
        logger.info("QA检索系统初始化成功")
        
except Exception as e:
    logger.error(f"QA检索系统初始化失败: {e}")
    qa_system_available = False

# MCP工具
@mcp.tool()
def get_current_date() -> str:
    """获取当前系统日期
    
    这个工具用于获取当前的系统日期，返回中文格式的日期字符串。
    
    📅 功能说明：
    - 获取实时的当前日期
    - 返回格式：YYYY年MM月DD日（如：2024年12月19日）
    - 基于系统本地时间
    
    🎯 适用场景：
    - 医疗记录需要当前日期
    - 预约挂号时间参考
    - 病历记录时间戳
    - 用药开始日期标记
    
    Returns:
        str: 当前日期，格式为"YYYY年MM月DD日"
    
    Example:
        调用该工具将返回类似"2024年12月19日"的日期字符串
    """
    result = datetime.now().strftime("%Y年%m月%d日")
    logger.info(f"get_current_date called, returning: {result}")
    return result


@mcp.tool()
def get_current_time() -> str:
    """获取当前系统时间
    
    这个工具用于获取当前的系统时间，返回24小时制格式的时间字符串。
    
    ⏰ 功能说明：
    - 获取实时的当前时间
    - 返回格式：HH:MM:SS（如：14:30:25）
    - 使用24小时制
    - 基于系统本地时间
    
    🎯 适用场景：
    - 医疗操作时间记录
    - 用药时间提醒
    - 检查时间标记
    - 急诊时间记录
    - 手术时间记录
    
    Returns:
        str: 当前时间，格式为"HH:MM:SS"
    
    Example:
        调用该工具将返回类似"14:30:25"的时间字符串
    """
    result = datetime.now().strftime("%H:%M:%S")
    logger.info(f"get_current_time called, returning: {result}")
    return result


@mcp.tool()
def medical_qa_search(question: str, image_analysis: str = "") -> str:
    """专业医学知识库检索问答工具（支持图像分析增强RAG）
    
    这是一个基于先进AI技术的医学知识检索系统，能够从专业医学知识库中检索相关信息并生成准确的医学答案。
    现在支持基于医疗图像分析结果的增强检索功能。
    
    🔬 技术特点：
    - 使用HyDE（假设性文档嵌入）技术提高检索精度
    - 结合QA问答对和原始医学文档进行检索
    - 使用RankGPT重排序算法优化结果相关性
    - 基于阿里云Qwen模型生成专业答案
    - 🆕 支持医疗图像分析结果增强RAG检索
    
    📚 知识库内容：
    - 疾病诊断指南
    - 治疗方案和用药指导
    - 症状分析和鉴别诊断
    - 医学术语和概念解释
    - 预防保健知识
    
    Args:
        question (str): 医学相关问题，支持中文自然语言提问
            
            问题类型示例：
            - 疾病症状："糖尿病有什么症状？"
            - 治疗方法："高血压的治疗方法有哪些？"
            - 诊断指南："如何诊断冠心病？"
            - 用药指导："阿司匹林的用法用量是什么？"
            - 预防保健："如何预防心脑血管疾病？"
            - 术语解释："什么是心房颤动？"
            - 检查项目："心电图能检查出什么问题？"
            
        image_analysis (str, optional): 医疗图像分析结果，用于增强RAG检索
            - 来源于X光、CT、MRI、心电图、血检报告等医疗影像的AI分析
            - 包含检查类型、异常发现、关键指标等医学信息
            - 系统会从中提取医学关键词来增强检索效果
    
    Returns:
        str: 基于医学知识库检索的详细专业答案
            - 内容准确可靠，来源于权威医学资料
            - 结构化组织信息，易于理解
            - 包含症状、诊断、治疗等全方位信息
            - 使用专业医学术语，同时兼顾通俗易懂
            - 🆕 如有图像分析，会整合影像信息与文献知识
    
    ⚠️ 重要提醒：
    - 本工具提供的信息仅供医学科普和参考
    - 不能替代专业医生的诊断和治疗建议
    - 如有健康问题，请及时就医咨询专业医生
    - 用药请遵医嘱，不可自行诊断用药
    
    Example:
        question = "介绍一下食管癌的症状"
        image_analysis = "CT检查显示食管壁增厚，管腔狭窄..."
        返回: 基于文献和影像的综合食管癌症状分析
    """
    if not qa_system_available:
        error_msg = "QA检索系统不可用，请检查DASHSCOPE_API_KEY环境变量或相关依赖"
        logger.error(error_msg)
        return error_msg
    
    try:
        # 强制将输入参数转换为字符串，避免模型报 contents 类型错误
        try:
            if not isinstance(question, str):
                import json as _json
                question = _json.dumps(question, ensure_ascii=False)
        except Exception:
            question = str(question)

        try:
            if image_analysis is None:
                image_analysis = ""
            elif not isinstance(image_analysis, str):
                import json as _json
                image_analysis = _json.dumps(image_analysis, ensure_ascii=False)
        except Exception:
            image_analysis = str(image_analysis)

        logger.info(f"medical_qa_search called with question: {question}")
        logger.info(f"📋 参数状态检查:")
        try:
            logger.info(f"  - question: '{question}' (长度: {len(question)})")
        except Exception:
            logger.info("  - question: <无法计算长度>")
        try:
            logger.info(f"  - image_analysis: {'存在' if image_analysis else '空'} (长度: {len(image_analysis)}, 类型: {type(image_analysis)})")
            logger.info(f"  - image_analysis内容预览: '{image_analysis[:100]}...' " if len(image_analysis) > 100 else f"  - image_analysis完整内容: '{image_analysis}'")
        except Exception:
            logger.info("  - image_analysis: <无法计算或非字符串类型>")
        
        # 简化查询处理：直接使用图像分析内容，跳过关键词提取AI
        enhanced_question = question
        
        if image_analysis and image_analysis.strip():
            logger.info(f"🖼️ 检测到图像分析内容，直接整合到查询中")
            # 直接将图像分析内容添加到查询中，不调用额外的关键词提取AI
            enhanced_question = f"{question} [医疗图像分析: {image_analysis[:500]}]"  # 限制长度避免过长
            logger.info(f"⚡ 快速整合查询完成: {enhanced_question[:100]}...")
        else:
            logger.info(f"⚠️ 无图像分析内容，使用标准RAG检索")
        
        # 增强的RAG缓存系统
        import hashlib
        import time
        query_hash = hashlib.md5(f"{enhanced_question}_{image_analysis}".encode('utf-8')).hexdigest()
        
        # 初始化全局RAG缓存
        if not hasattr(medical_qa_search, '_rag_cache'):
            medical_qa_search._rag_cache = {}
        
        # 检查缓存
        rag_answer = None
        if query_hash in medical_qa_search._rag_cache:
            cache_entry = medical_qa_search._rag_cache[query_hash]
            if time.time() - cache_entry['timestamp'] < 7200:  # 延长到2小时缓存
                logger.info(f"⚡ RAG缓存命中: {question[:50]}...")
                rag_answer = cache_entry['answer']
                # 更新访问时间
                cache_entry['last_accessed'] = time.time()
            else:
                del medical_qa_search._rag_cache[query_hash]
        
        # 如果没有缓存，执行RAG检索
        if rag_answer is None:
            response = qa_graph.invoke({"question": enhanced_question})
            answer_obj = response.get("answer")
            try:
                rag_answer_raw = getattr(answer_obj, "content", answer_obj)
                if isinstance(rag_answer_raw, list):
                    rag_answer = "".join([
                        part.get("text", str(part)) if isinstance(part, dict) else str(part)
                        for part in rag_answer_raw
                    ])
                elif isinstance(rag_answer_raw, str):
                    rag_answer = rag_answer_raw
                else:
                    import json as _json
                    rag_answer = _json.dumps(rag_answer_raw, ensure_ascii=False)
            except Exception:
                rag_answer = str(getattr(answer_obj, "content", answer_obj))
            
            # 缓存结果
            medical_qa_search._rag_cache[query_hash] = {
                'answer': rag_answer,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'question_preview': question[:100]
            }
            
            # 智能缓存清理：优先清理最少访问的条目
            if len(medical_qa_search._rag_cache) > 100:  # 增加缓存容量
                # 清理过期条目
                current_time = time.time()
                expired_keys = [k for k, v in medical_qa_search._rag_cache.items() 
                              if current_time - v['timestamp'] > 7200]
                for k in expired_keys:
                    del medical_qa_search._rag_cache[k]
                
                # 如果还是太多，清理最少访问的
                if len(medical_qa_search._rag_cache) > 100:
                    oldest_key = min(medical_qa_search._rag_cache.keys(), 
                                   key=lambda k: medical_qa_search._rag_cache[k].get('last_accessed', 0))
                    del medical_qa_search._rag_cache[oldest_key]
            
            logger.info(f"💾 RAG结果已缓存: {question[:50]}... (缓存大小: {len(medical_qa_search._rag_cache)})")
        
        # 简化处理：直接返回RAG结果，跳过图像信息整合AI
        if image_analysis and image_analysis.strip():
            logger.info(f"⚡ 图像增强RAG查询完成，直接返回结果（跳过整合AI以提升速度）")
            # 在RAG答案前添加图像分析摘要，但不调用额外的整合AI
            image_summary = image_analysis[:200] + "..." if len(image_analysis) > 200 else image_analysis
            final_answer = f"【医疗图像分析】\n{image_summary}\n\n【专业医学知识】\n{rag_answer}\n\n⚠️ 以上信息仅供参考，请咨询专业医生进行确诊。"
            return final_answer
        else:
            logger.info(f"✅ 标准RAG查询完成")
            return rag_answer
        
    except Exception as e:
        error_msg = f"医学问答检索过程中发生错误：{str(e)}"
        logger.error(f"medical_qa_search error: {error_msg}")
        return error_msg


@mcp.tool()
def get_hospital_info(cnName: str = "", address: str = "", enName: str = "", cnShort: str = "", hospitalType: str = "", level: str = "", ownership: str = "", pageSize: int = 10) -> str:
    """全国医院信息查询工具
    
    这是一个专业的医院信息查询系统，连接全国医院数据库，提供详细的医院信息查询服务。
    支持多维度搜索条件，帮助用户快速找到合适的医疗机构。
    
    🏥 查询功能：
    - 支持精确匹配和模糊搜索
    - 多条件组合查询
    - 实时获取最新医院信息
    - 提供详细医院资料
    
    📊 返回信息包括：
    - 医院基本信息（名称、地址、电话）
    - 医院规模（床位数、员工数）
    - 医院等级和性质
    - 建院时间和简介
    - 官方网站
    
    Args:
        cnName (str, optional): 医院中文名称
            - 支持模糊搜索，如输入"人民医院"可找到所有包含此关键词的医院
            - 示例："北京协和医院"、"上海第一人民医院"
            - 留空表示不限制医院名称
            
        address (str, optional): 医院地址或所在地区
            - 支持省市区县级搜索
            - 示例："北京市"、"上海市浦东新区"、"广州"
            - 可用于查找特定地区的医院
            
        enName (str, optional): 医院英文名称
            - 用于查询有英文名称的医院
            - 示例："Peking Union Medical College Hospital"
            - 主要用于查询国际化医院或知名医院
            
        cnShort (str, optional): 医院中文简称或别名
            - 医院的简称或常用别名
            - 示例："协和"、"301医院"、"华西"
            - 方便使用医院常用简称查询
            
        hospitalType (str, optional): 医院类型
            - 可选值："综合性医院"、"专科医院"、"中医医院"、"妇幼保健院"等
            - 帮助查找特定类型的医疗机构
            - 示例：查找"专科医院"可找到眼科医院、心血管医院等
            
        level (str, optional): 医院等级
            - 可选值："三级甲等"、"三级乙等"、"二级甲等"、"二级乙等"、"一级甲等"等
            - 用于查找特定等级的医院
            - 示例："三级甲等"查找最高等级医院
            
        ownership (str, optional): 医院性质
            - 可选值："公立"、"私立"、"民营"、"合资"等
            - 用于区分医院的所有制性质
            - 示例："公立"查找公立医院
            
        pageSize (int, optional): 每页显示的医院数量
            - 默认值：10
            - 范围：1-50
            - 控制返回结果的数量，避免信息过载
    
    Returns:
        str: 格式化的医院信息列表
            包含以下详细信息：
            - 医院中文名称和英文名称
            - 医院地址和联系电话
            - 医院类型、等级和性质
            - 床位数和员工数
            - 建院时间和医院简介
            - 官方网站
            
            如果未找到匹配的医院，会返回相应的提示信息。
    
    🎯 使用场景：
    - 就医推荐：根据地区和专科查找合适医院
    - 医院对比：比较不同医院的规模和等级
    - 转院参考：查找上级医院或专科医院
    - 医疗资源调研：了解某地区医疗资源分布
    
    💡 查询技巧：
    1. 单条件查询：只填写一个参数进行宽泛搜索
    2. 组合查询：结合多个条件精确查找
    3. 地区查询：使用address参数查找本地医院
    4. 等级筛选：使用level参数查找高等级医院
    
    Example:
        # 查找北京的三级甲等医院
        cnName="", address="北京", level="三级甲等"
        
        # 查找协和医院的详细信息
        cnName="协和医院"
        
        # 查找上海的专科医院
        address="上海", hospitalType="专科医院"
    """
    try:
        # API URL
        url = "http://localhost:48080/admin-api/datamanagement/hospital/page"
        
        # 请求参数
        params = {
            "pageSize": pageSize
        }
        
        # 添加所有搜索条件到参数中
        if cnName:
            params["cnName"] = cnName
        if address:
            params["address"] = address
        if enName:
            params["enName"] = enName
        if cnShort:
            params["cnShort"] = cnShort
        if hospitalType:
            params["hospitalType"] = hospitalType
        if level:
            params["level"] = level
        if ownership:
            params["ownership"] = ownership
        
        # 请求头
        headers = {
            "tenant-id": "1",
            "Content-Type": "application/json"
        }
        
        logger.info(f"get_hospital_info called with cnName='{cnName}', address='{address}', enName='{enName}', cnShort='{cnShort}', hospitalType='{hospitalType}', level='{level}', ownership='{ownership}', pageSize={pageSize}")
        
        # 发送HTTP请求
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析响应数据
        data = response.json()
        
        if data.get("code") == 0:
            hospital_list = data.get("data", {}).get("list", [])
            total = data.get("data", {}).get("total", 0)
            
            if hospital_list:
                # 格式化医院信息
                search_conditions = []
                if cnName:
                    search_conditions.append(f"医院名称: {cnName}")
                if address:
                    search_conditions.append(f"地址: {address}")
                if enName:
                    search_conditions.append(f"英文名称: {enName}")
                if cnShort:
                    search_conditions.append(f"中文简称: {cnShort}")
                if hospitalType:
                    search_conditions.append(f"医院类型: {hospitalType}")
                if level:
                    search_conditions.append(f"医院等级: {level}")
                if ownership:
                    search_conditions.append(f"医院性质: {ownership}")
                
                result_info = f"找到 {total} 家医院"
                if search_conditions:
                    search_info = "、".join(search_conditions)
                    result_info += f"（搜索条件：{search_info}）"
                result_info += "：\n\n"
                
                for hospital in hospital_list:
                    result_info += f"医院中文名称：{hospital.get('cnName', '未知')}\n"
                    result_info += f"医院中文缩写名称：{hospital.get('cnShort', '未知')}\n"
                    result_info += f"医院英文名称：{hospital.get('enName', '未知')}\n"
                    result_info += f"医院类型：{hospital.get('hospitalType', '未知')}\n"
                    result_info += f"医院等级：{hospital.get('level', '未知')}\n"
                    result_info += f"所有制：{hospital.get('ownership', '未知')}\n"
                    result_info += f"地址：{hospital.get('address', '未知')}\n"
                    result_info += f"电话：{hospital.get('phone', '未提供')}\n"
                    result_info += f"床位数：{hospital.get('bedCount', 0)}\n"
                    result_info += f"员工数：{hospital.get('staffCount', 0)}\n"
                    if hospital.get('establishedYear'):
                        result_info += f"建院时间：{hospital.get('establishedYear')}年\n"
                    if hospital.get('introduction'):
                        intro = hospital.get('introduction', '')
                        result_info += f"医院简介：{intro}...\n"
                    result_info += f"网站：{hospital.get('website', '未提供')}\n"
                    result_info += "\n" + "="*50 + "\n\n"
                
                logger.info(f"get_hospital_info success: found {total} hospitals")
                return result_info
            else:
                search_conditions = []
                if cnName:
                    search_conditions.append(f"医院名称: {cnName}")
                if address:
                    search_conditions.append(f"地址: {address}")
                if enName:
                    search_conditions.append(f"英文名称: {enName}")
                if cnShort:
                    search_conditions.append(f"中文简称: {cnShort}")
                if hospitalType:
                    search_conditions.append(f"医院类型: {hospitalType}")
                if level:
                    search_conditions.append(f"医院等级: {level}")
                if ownership:
                    search_conditions.append(f"医院性质: {ownership}")
                search_text = "、".join(search_conditions) if search_conditions else "无特定条件"
                
                result = f"未找到符合条件的医院（搜索条件：{search_text}）"
                logger.info(f"get_hospital_info: no hospitals found")
                return result
        else:
            error_msg = f"API返回错误：{data.get('msg', '未知错误')}"
            logger.error(f"get_hospital_info API error: {error_msg}")
            return error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"网络请求失败：{str(e)}"
        logger.error(f"get_hospital_info network error: {error_msg}")
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败：{str(e)}"
        logger.error(f"get_hospital_info JSON error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"获取医院信息时发生错误：{str(e)}"
        logger.error(f"get_hospital_info unexpected error: {error_msg}")
        return error_msg

# 启动MCP服务器
if __name__ == "__main__":

    # 启动服务器，绑定到所有接口以避免网络问题
    logger.info("正在启动MCP服务器...")
    try:
        mcp.run(
            transport="sse",
            host="0.0.0.0",  # 绑定到所有接口
            port=18002  # 使用更高端口避免冲突
        )
    except Exception as e:
        logger.error(f"MCP服务器启动失败: {e}")
        sys.exit(1)
