import os
import glob
import re
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import requests


# =============================================================================
# QA生成相关配置和提示模板
# =============================================================================
load_dotenv()

# 检查阿里云API密钥
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key:
    print("❌ 阿里云DASHSCOPE_API_KEY未设置！")
    print("请设置环境变量：")
    print("  DASHSCOPE_API_KEY = 您的阿里云DashScope API Key")
    print("\n获取方式：")
    print("1. 访问 https://dashscope.console.aliyun.com/")
    print("2. 创建或选择应用，获取API Key")
    print("3. 设置环境变量或在.env文件中配置")
    print("\n示例配置(.env文件)：")
    print("DASHSCOPE_API_KEY=sk-***********")
    import sys
    sys.exit(1)

print(f"✅ 阿里云API密钥配置检查通过")
print(f"   API Key: {dashscope_api_key[:8]}***（已脱敏显示）")

# 初始化阿里云DashScope模型
llm_for_qa_generation = ChatOpenAI(
    model="qwen-max",
    temperature=0.1,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=dashscope_api_key,
)

# QA生成提示模板
QA_GENERATION_PROMPT = PromptTemplate(
    template="""你是一位专业的医学教育专家。请基于以下医学文档内容，生成高质量的问答对，用于医学教育和临床参考。

文档内容：
{content}

请根据上述内容生成3-5个问答对，要求：
1. 问题应该是临床实践中常见的、有实际意义的问题
2. 答案必须基于提供的文档内容，准确且详细
3. 问题类型要多样化：诊断、治疗、症状、检查、预后等
4. 答案要专业但易懂，适合医学学习和临床参考
5. 避免生成过于简单或过于复杂的问题

请按以下格式输出：
问题1：[问题内容]
答案1：[答案内容]

问题2：[问题内容]
答案2：[答案内容]

问题3：[问题内容]
答案3：[答案内容]

[继续按需生成更多QA对...]
""",
    input_variables=["content"]
)

# 预定义的目标QA集合（常见医学问题）
TARGET_QA_PAIRS = [
    {
        "question": "冠心病的典型症状有哪些？",
        "answer": "冠心病的典型症状主要包括：1) 胸痛或胸闷：典型的心绞痛表现为胸骨后压榨性疼痛，常在活动时出现，休息后缓解；2) 活动后气促：运动耐力下降，轻度活动即感气短；3) 心悸：心率不齐或心动过速的感觉；4) 疲劳乏力：日常活动能力下降；5) 放射痛：疼痛可放射至左肩、左臂、颈部或下颌。需要注意的是，部分患者可能出现不典型症状，如上腹痛、消化不良等。",
        "category": "症状识别"
    },
    {
        "question": "如何诊断稳定型心绞痛？",
        "answer": "稳定型心绞痛的诊断主要基于：1) 临床症状：典型的胸痛特点（部位、性质、诱发因素、持续时间、缓解方式）；2) 心电图检查：静息心电图可能正常，运动负荷试验可诱发ST段改变；3) 超声心动图：评估心脏结构和功能；4) 冠状动脉造影：金标准，直接显示冠脉狭窄情况；5) 冠脉CT：无创性检查，适用于中低风险患者；6) 核医学检查：心肌灌注显像等。诊断需要综合临床表现和辅助检查结果。",
        "category": "诊断方法"
    },
    {
        "question": "冠心病的药物治疗原则是什么？",
        "answer": "冠心病的药物治疗遵循以下原则：1) 抗血小板治疗：阿司匹林为基础，急性期可联合氯吡格雷；2) 他汀类药物：降脂稳斑，目标LDL-C<1.8mmol/L；3) ACEI/ARB：改善预后，特别适用于合并糖尿病或心功能不全患者；4) β受体阻滞剂：减慢心率，减少心肌耗氧量；5) 硝酸酯类：缓解心绞痛症状；6) 钙通道阻滞剂：适用于变异性心绞痛或不耐受β受体阻滞剂的患者。治疗需要个体化，根据患者具体情况调整用药方案。",
        "category": "治疗方案"
    },
    {
        "question": "冠状动脉造影的适应症有哪些？",
        "answer": "冠状动脉造影的适应症包括：1) 急性冠脉综合征：急性心肌梗死、不稳定性心绞痛；2) 稳定性心绞痛药物治疗效果不佳；3) 非侵入性检查提示存在显著心肌缺血；4) 心脏猝死复苏后；5) 术前评估：心脏手术前的冠脉评估；6) 怀疑冠脉异常：先天性冠脉异常、冠脉瘘等；7) 心功能不全原因不明需要排除缺血性心肌病；8) 职业需要：飞行员、司机等特殊职业的冠脉评估。造影前需要充分评估风险效益比。",
        "category": "检查适应症"
    },
    {
        "question": "急性心肌梗死的急救处理原则是什么？",
        "answer": "急性心肌梗死的急救处理原则：1) 立即评估：12导联心电图，评估血流动力学状态；2) 迅速再灌注：发病12小时内首选PCI，不具备条件时溶栓治疗；3) 抗栓治疗：双重抗血小板+抗凝治疗；4) 对症处理：镇痛（吗啡）、吸氧、硝酸甘油等；5) 并发症监测：心律失常、心力衰竭、机械并发症等；6) 早期风险分层：评估梗死面积、左心功能等；7) 二级预防：他汀、ACEI/ARB、β受体阻滞剂等。关键是'时间就是心肌'，尽快恢复血流。",
        "category": "急救处理"
    },
    {
        "question": "冠心病患者的生活方式管理包括哪些方面？",
        "answer": "冠心病患者的生活方式管理包括：1) 戒烟限酒：完全戒烟，限制酒精摄入；2) 饮食控制：低盐低脂饮食，增加蔬菜水果摄入，控制总热量；3) 规律运动：有氧运动为主，每周150分钟中等强度运动；4) 体重控制：维持理想体重，BMI<24kg/m²；5) 血压管理：目标血压<130/80mmHg；6) 血糖控制：糖尿病患者HbA1c<7%；7) 血脂管理：LDL-C<1.8mmol/L；8) 压力管理：保持良好心态，避免过度紧张；9) 规律作息：充足睡眠，避免熬夜。生活方式干预是冠心病防治的基础。",
        "category": "生活管理"
    }
]


# =============================================================================
# 文本向量化处理
# =============================================================================

class DashScopeEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-v4",
                 api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                 request_timeout_seconds: int = 30,
                 sleep_between_requests_seconds: float = 0.05):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = request_timeout_seconds
        self.sleep_seconds = sleep_between_requests_seconds

    def embed_query(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        raise ValueError(f"无效的Embedding响应: {data}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            try:
                vector = self.embed_query(text)
            except Exception:
                vector = [0.0] * 1024
            embeddings.append(vector)
            if self.sleep_seconds and self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        return embeddings

# =============================================================================
# 通用文本处理和验证函数
# =============================================================================

def clean_text(text: str) -> str:
    """清理文本，移除无用字符和格式"""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除单独的特殊字符
    text = re.sub(r'\s[^\w\u4e00-\u9fff]{1,3}\s', ' ', text)
    # 移除过短的英文单词片段（可能是OCR错误）
    text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
    # 移除连续的特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。；：！？、""''（）【】《》]{2,}', ' ', text)
    # 再次清理多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_valid_content(content: str, min_length: int = 50) -> bool:
    """判断内容是否有效"""
    # 先清理一下内容再判断
    cleaned = clean_text(content)

    if len(cleaned) < min_length:
        return False

    # 计算中文字符比例
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned))
    total_chars = len(cleaned)
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

    # 计算有意义单词的比例
    words = cleaned.split()
    meaningful_words = [w for w in words if len(w) > 2 and re.search(r'[a-zA-Z\u4e00-\u9fff]', w)]
    word_ratio = len(meaningful_words) / len(words) if len(words) > 0 else 0

    # 中文内容或高质量英文内容都认为有效
    return chinese_ratio > 0.1 or (word_ratio > 0.7 and len(meaningful_words) > 5)


def validate_document_before_embedding(doc: Document, min_length: int = 100) -> bool:
    """在发送到embeddings API之前进行最终验证"""
    content = doc.page_content.strip()

    # 基本长度检查
    if len(content) < min_length:
        print(f"    文档内容过短({len(content)}字符)，跳过")
        return False

    # 检查是否只包含空白或特殊字符
    meaningful_chars = re.sub(r'[\s\W]', '', content)
    if len(meaningful_chars) < min_length // 2:
        print(f"    文档缺乏有意义内容，跳过")
        return False

    # 检查是否包含足够的中文或英文内容
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
    english_chars = len(re.findall(r'[a-zA-Z]', content))

    if chinese_chars < 20 and english_chars < 50:
        print(f"    文档语言内容不足，跳过")
        return False

    return True


# =============================================================================
# QA生成功能
# =============================================================================

def generate_qa_pairs_from_document(doc: Document) -> List[Document]:
    """从单个文档生成QA对"""
    qa_documents = []

    try:
        # 确保文档内容足够长且有效
        if len(doc.page_content) < 200 or not is_valid_content(doc.page_content, min_length=100):
            print(f"    文档内容不足，跳过QA生成")
            return []

        print(f"    正在为文档生成QA对...")

        # 调用LLM生成QA对
        prompt_input = {"content": doc.page_content[:4000]}  # 限制长度避免token超限
        response = llm_for_qa_generation.invoke(QA_GENERATION_PROMPT.format(**prompt_input))

        if not response or not response.content:
            print(f"    QA生成失败：无有效响应")
            return []

        # 解析生成的QA对
        qa_pairs = parse_generated_qa_pairs(response.content)

        if not qa_pairs:
            print(f"    QA解析失败：未找到有效QA对")
            return []

        print(f"    成功生成 {len(qa_pairs)} 个QA对")

        # 为每个QA对创建Document对象
        for i, (question, answer) in enumerate(qa_pairs):
            # 创建QA文档内容
            qa_content = f"问题：{question}\n\n答案：{answer}"

            # 创建QA文档的元数据
            qa_metadata = {
                **doc.metadata,  # 继承原文档元数据
                'content_type': 'generated_qa',
                'qa_pair_index': i + 1,
                'question': question,
                'answer': answer,
                'source_content_length': len(doc.page_content),
                'generation_method': 'llm_generated',
                'file_type': 'qa_pair'
            }

            qa_doc = Document(
                page_content=qa_content,
                metadata=qa_metadata
            )

            qa_documents.append(qa_doc)

    except Exception as e:
        print(f"    QA生成过程出错: {e}")

    return qa_documents


def parse_generated_qa_pairs(qa_text: str) -> List[Tuple[str, str]]:
    """解析LLM生成的QA文本"""
    qa_pairs = []

    try:
        # 使用正则表达式提取QA对
        # 匹配 "问题X：...答案X：..." 的模式
        pattern = r'问题\d*[：:]\s*([^\n]+)\s*\n\s*答案\d*[：:]\s*([^问]+?)(?=问题\d*[：:]|$)'
        matches = re.findall(pattern, qa_text, re.DOTALL)

        for question, answer in matches:
            question = question.strip()
            answer = answer.strip()

            # 验证QA对的质量
            if (len(question) > 10 and len(answer) > 20 and
                    question.endswith(('？', '?', '吗', '呢')) and
                    is_valid_content(answer, min_length=20)):
                qa_pairs.append((question, answer))

        # 如果正则表达式匹配失败，尝试其他解析方法
        if not qa_pairs:
            lines = qa_text.split('\n')
            current_question = None
            current_answer = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(('问题', '题目')) and ('：' in line or ':' in line):
                    # 保存上一个QA对
                    if current_question and current_answer:
                        qa_pairs.append((current_question, current_answer.strip()))

                    # 开始新的问题
                    current_question = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                    current_answer = ""

                elif line.startswith(('答案', '回答')) and ('：' in line or ':' in line):
                    current_answer = line.split('：', 1)[-1].split(':', 1)[-1].strip()

                elif current_question and current_answer:
                    # 继续收集答案内容
                    current_answer += " " + line

            # 保存最后一个QA对
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer.strip()))

    except Exception as e:
        print(f"    QA解析出错: {e}")

    return qa_pairs


def create_target_qa_documents() -> List[Document]:
    """创建预定义的目标QA文档"""
    qa_documents = []

    print(f"🎯 创建预定义目标QA文档集合...")

    for i, qa_pair in enumerate(TARGET_QA_PAIRS):
        # 创建QA文档内容
        qa_content = f"问题：{qa_pair['question']}\n\n答案：{qa_pair['answer']}"

        # 创建元数据
        qa_metadata = {
            'source': 'predefined_target_qa',
            'source_file': 'target_qa_collection',
            'content_type': 'target_qa',
            'qa_pair_index': i + 1,
            'question': qa_pair['question'],
            'answer': qa_pair['answer'],
            'category': qa_pair['category'],
            'generation_method': 'predefined',
            'file_type': 'qa_pair',
            'content_length': len(qa_content)
        }

        qa_doc = Document(
            page_content=qa_content,
            metadata=qa_metadata
        )

        qa_documents.append(qa_doc)

    print(f"✅ 成功创建 {len(qa_documents)} 个目标QA文档")
    return qa_documents


# =============================================================================
# PDF处理相关函数
# =============================================================================

def detect_chapter_patterns(text: str) -> List[str]:
    """检测文本中的章节标题模式"""
    patterns = [
        # 数字编号模式 (1. 2. 3. 或 1、2、3、)
        r'^[\s]*(\d+[.、][\s]*[^\n]{5,50})$',
        # 多级编号 (1.1 1.2 2.1 等)
        r'^[\s]*(\d+\.\d+[\s]*[^\n]{5,50})$',
        # 中文数字 (一、二、三、)
        r'^[\s]*([一二三四五六七八九十百千万]+[、.][\s]*[^\n]{5,50})$',
        # 括号编号 ((1) (2) (3))
        r'^[\s]*(\([一二三四五六七八九十\d]+\)[\s]*[^\n]{5,50})$',
        # 英文罗马数字 (I. II. III.)
        r'^[\s]*([IVX]+[.][\s]*[^\n]{5,50})$',
        # 其他可能的标题格式（全大写或包含关键词）
        r'^[\s]*([^\n]{5,30}[诊断|治疗|管理|指南|建议|原则|方法|策略|分析])[\s]*$',
    ]

    detected_patterns = []
    lines = text.split('\n')

    for pattern in patterns:
        matches = []
        for line in lines:
            if re.match(pattern, line.strip(), re.MULTILINE):
                matches.append(line.strip())
        if len(matches) >= 2:  # 至少要有2个匹配才认为是有效模式
            detected_patterns.append(pattern)

    return detected_patterns


def split_by_chapters(text: str, metadata: dict) -> List[Document]:
    """按章节切分文档"""
    # 检测章节模式
    patterns = detect_chapter_patterns(text)

    if not patterns:
        print(f"    未检测到明显的章节模式，使用传统切分方式")
        # 如果没有检测到章节模式，回退到传统切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", "。", "；", "！", "？", ".", "!", "?", " ", ""]
        )
        temp_doc = Document(page_content=text, metadata=metadata)
        return text_splitter.split_documents([temp_doc])

    print(f"    检测到 {len(patterns)} 种章节模式，按章节切分")

    # 使用第一个最有效的模式进行切分
    primary_pattern = patterns[0]
    lines = text.split('\n')

    chapters = []
    current_chapter = []
    current_title = "引言"

    for line in lines:
        # 检查是否是章节标题
        if re.match(primary_pattern, line.strip(), re.MULTILINE):
            # 保存上一章节
            if current_chapter:
                chapter_content = '\n'.join(current_chapter).strip()
                if is_valid_content(chapter_content, min_length=100):
                    chapter_doc = Document(
                        page_content=chapter_content,
                        metadata={
                            **metadata,
                            'chapter_title': current_title,
                            'split_method': 'chapter_based',
                            'file_type': 'pdf'
                        }
                    )
                    chapters.append(chapter_doc)

            # 开始新章节
            current_title = line.strip()
            current_chapter = [line]
        else:
            current_chapter.append(line)

    # 处理最后一章
    if current_chapter:
        chapter_content = '\n'.join(current_chapter).strip()
        if is_valid_content(chapter_content, min_length=100):
            chapter_doc = Document(
                page_content=chapter_content,
                metadata={
                    **metadata,
                    'chapter_title': current_title,
                    'split_method': 'chapter_based',
                    'file_type': 'pdf'
                }
            )
            chapters.append(chapter_doc)

    # 如果章节太大，进一步细分
    final_chapters = []
    for chapter in chapters:
        if len(chapter.page_content) > 3000:  # 如果章节过大
            print(
                f"      章节 '{chapter.metadata['chapter_title'][:20]}...' 过大({len(chapter.page_content)}字符)，进一步细分")
            # 在章节内使用传统方法进一步切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", "。", "；", "！", "？", ".", "!", "?", " ", ""]
            )
            sub_docs = text_splitter.split_documents([chapter])

            # 为子文档添加章节信息
            for i, sub_doc in enumerate(sub_docs):
                sub_doc.metadata.update({
                    'chapter_title': chapter.metadata['chapter_title'],
                    'sub_section': i + 1,
                    'split_method': 'chapter_then_size',
                    'file_type': 'pdf'
                })
            final_chapters.extend(sub_docs)
        else:
            final_chapters.append(chapter)

    return final_chapters


def load_pdf_with_multiple_strategies(pdf_path: str) -> List[Document]:
    """使用多种策略加载PDF，选择最佳结果"""
    documents = []

    # 策略1：尝试PyMuPDF（通常对文本PDF效果更好）
    try:
        print(f"  尝试策略1：PyMuPDF")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        if docs:
            combined_text = "\n".join([doc.page_content for doc in docs])
            cleaned_text = clean_text(combined_text)
            if is_valid_content(cleaned_text, min_length=100):
                print(f"  策略1成功，提取了 {len(cleaned_text)} 个字符")
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": pdf_path, "extraction_method": "pymupdf", "file_type": "pdf"}
                ))
                return documents
    except Exception as e:
        print(f"  策略1失败: {e}")

    # 策略2：使用UnstructuredPDFLoader的auto策略
    try:
        print(f"  尝试策略2：Unstructured auto")
        loader = UnstructuredPDFLoader(
            file_path=pdf_path,
            mode="elements",
            strategy="auto",  # 改为auto，让它自动选择最佳策略
        )
        docs = loader.load()
        if docs:
            combined_text = "\n".join([doc.page_content for doc in docs])
            cleaned_text = clean_text(combined_text)
            if is_valid_content(cleaned_text, min_length=100):
                print(f"  策略2成功，提取了 {len(cleaned_text)} 个字符")
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": pdf_path, "extraction_method": "unstructured_auto", "file_type": "pdf"}
                ))
                return documents
    except Exception as e:
        print(f"  策略2失败: {e}")

    # 策略3：最后尝试OCR（作为备选）
    try:
        print(f"  尝试策略3：OCR")
        loader = UnstructuredPDFLoader(
            file_path=pdf_path,
            mode="elements",
            strategy="ocr_only",
        )
        docs = loader.load()
        if docs:
            combined_text = "\n".join([doc.page_content for doc in docs])
            cleaned_text = clean_text(combined_text)
            if is_valid_content(cleaned_text, min_length=50):  # OCR的要求降低一些
                print(f"  策略3成功，提取了 {len(cleaned_text)} 个字符")
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": pdf_path, "extraction_method": "ocr", "file_type": "pdf"}
                ))
                return documents
    except Exception as e:
        print(f"  策略3失败: {e}")

    print(f"  所有策略都失败")
    return []


# =============================================================================
# Excel处理相关函数
# =============================================================================

def analyze_excel_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """分析Excel表格结构，识别数据类型和模式"""
    structure_info = {
        'column_types': {},
        'numeric_columns': [],
        'text_columns': [],
        'date_columns': [],
        'categorical_columns': [],
        'has_headers': True,
        'row_count': len(df),
        'col_count': len(df.columns),
        'empty_cells_ratio': 0,
        'data_density': 0
    }

    # 分析每列的数据类型和特征
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        # 检测数据类型
        if pd.api.types.is_numeric_dtype(col_data):
            structure_info['numeric_columns'].append(col)
            structure_info['column_types'][col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            structure_info['date_columns'].append(col)
            structure_info['column_types'][col] = 'date'
        else:
            # 文本列进一步分析
            unique_ratio = len(col_data.unique()) / len(col_data)
            if unique_ratio < 0.5:  # 重复值较多，可能是分类列
                structure_info['categorical_columns'].append(col)
                structure_info['column_types'][col] = 'categorical'
            else:
                structure_info['text_columns'].append(col)
                structure_info['column_types'][col] = 'text'

    # 计算数据密度
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.count().sum()
    structure_info['data_density'] = non_null_cells / total_cells if total_cells > 0 else 0
    structure_info['empty_cells_ratio'] = 1 - structure_info['data_density']

    return structure_info


def create_table_summary(df: pd.DataFrame, sheet_name: str, structure_info: Dict[str, Any]) -> str:
    """创建表格的结构化摘要"""
    summary_parts = [
        f"=== 表格摘要：{sheet_name} ===",
        f"数据维度：{structure_info['row_count']}行 × {structure_info['col_count']}列",
        f"数据完整性：{structure_info['data_density']:.1%} （{structure_info['row_count'] * structure_info['col_count'] - df.isnull().sum().sum()}/{structure_info['row_count'] * structure_info['col_count']}个非空单元格）"
    ]

    # 添加列信息
    if structure_info['numeric_columns']:
        summary_parts.append(
            f"数值列 ({len(structure_info['numeric_columns'])}个)：{', '.join(structure_info['numeric_columns'])}")

    if structure_info['categorical_columns']:
        summary_parts.append(
            f"分类列 ({len(structure_info['categorical_columns'])}个)：{', '.join(structure_info['categorical_columns'])}")

    if structure_info['text_columns']:
        summary_parts.append(
            f"文本列 ({len(structure_info['text_columns'])}个)：{', '.join(structure_info['text_columns'])}")

    if structure_info['date_columns']:
        summary_parts.append(
            f"日期列 ({len(structure_info['date_columns'])}个)：{', '.join(structure_info['date_columns'])}")

    # 添加数据统计
    numeric_stats = []
    for col in structure_info['numeric_columns'][:3]:  # 只显示前3个数值列的统计
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats = f"{col}：最小值{col_data.min():.2f}, 最大值{col_data.max():.2f}, 平均值{col_data.mean():.2f}"
            numeric_stats.append(stats)

    if numeric_stats:
        summary_parts.append("主要数值列统计：")
        summary_parts.extend([f"  {stat}" for stat in numeric_stats])

    return "\n".join(summary_parts)


def format_table_data_intelligently(df: pd.DataFrame, structure_info: Dict[str, Any], start_row: int = 0,
                                    max_rows: int = 20) -> str:
    """智能格式化表格数据，保持语义关系"""
    formatted_parts = []

    # 获取要处理的行范围
    end_row = min(start_row + max_rows, len(df))
    subset_df = df.iloc[start_row:end_row]

    formatted_parts.append(f"=== 数据记录 (第{start_row + 1}行到第{end_row}行) ===")

    for idx, (_, row) in enumerate(subset_df.iterrows(), start=start_row + 1):
        # 构建更自然的描述
        record_description = f"记录{idx}："

        # 优先显示重要的识别信息（通常是前几列或文本列）
        primary_fields = []
        secondary_fields = []

        for col_name, value in row.items():
            if pd.isna(value) or str(value).strip() == '':
                continue

            formatted_value = format_cell_value(value, structure_info['column_types'].get(col_name, 'text'))
            field_text = f"{col_name}为{formatted_value}"

            # 分类列和主要文本列作为主要字段
            if (col_name in structure_info.get('categorical_columns', []) or
                    col_name in structure_info.get('text_columns', [])[:2]):  # 前两个文本列
                primary_fields.append(field_text)
            else:
                secondary_fields.append(field_text)

        # 组合字段描述
        if primary_fields:
            record_description += "，".join(primary_fields)
            if secondary_fields:
                record_description += "；其他信息：" + "，".join(secondary_fields[:5])  # 限制显示数量
        else:
            record_description += "，".join(secondary_fields[:8])  # 如果没有主要字段，显示更多次要字段

        formatted_parts.append(record_description)

    return "\n".join(formatted_parts)


def format_cell_value(value: Any, col_type: str) -> str:
    """根据列类型智能格式化单元格值"""
    if pd.isna(value):
        return "空值"

    if col_type == 'numeric':
        if isinstance(value, (int, float)):
            if float(value).is_integer():
                return f"{int(value)}"
            else:
                return f"{float(value):.2f}".rstrip('0').rstrip('.')
    elif col_type == 'date':
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y年%m月%d日')
        elif isinstance(value, str):
            # 尝试解析日期字符串
            try:
                parsed_date = pd.to_datetime(value)
                return parsed_date.strftime('%Y年%m月%d日')
            except:
                pass

    # 默认转为字符串，并清理
    str_value = str(value).strip()
    # 移除过长的文本，保留重要信息
    if len(str_value) > 50:
        str_value = str_value[:47] + "..."

    return str_value


def create_column_relationship_text(df: pd.DataFrame, structure_info: Dict[str, Any]) -> str:
    """创建列之间关系的描述文本"""
    relationship_parts = ["=== 数据关系分析 ==="]

    # 分析分类列的值分布
    for col in structure_info.get('categorical_columns', [])[:3]:  # 最多分析3个分类列
        value_counts = df[col].value_counts().head(5)
        if len(value_counts) > 0:
            value_desc = "、".join([f"{val}({count}次)" for val, count in value_counts.items()])
            relationship_parts.append(f"{col}字段的主要值包括：{value_desc}")

    # 分析数值列的范围和分布
    for col in structure_info.get('numeric_columns', [])[:3]:  # 最多分析3个数值列
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q25, q75 = col_data.quantile([0.25, 0.75])
            relationship_parts.append(f"{col}字段数值分布：25%分位数为{q25:.2f}，75%分位数为{q75:.2f}")

    # 尝试发现列之间的关联（简单的相关性分析）
    numeric_cols = structure_info.get('numeric_columns', [])
    if len(numeric_cols) >= 2:
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, min(i + 3, len(numeric_cols))):  # 只检查临近的几列
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr = df[col1].corr(df[col2])
                if not pd.isna(corr) and abs(corr) > 0.5:
                    correlation_desc = "强正相关" if corr > 0.5 else "强负相关"
                    correlations.append(f"{col1}与{col2}存在{correlation_desc}(相关系数{corr:.2f})")

        if correlations:
            relationship_parts.append("发现的数据关联：")
            relationship_parts.extend([f"  {corr}" for corr in correlations])

    return "\n".join(relationship_parts)


def split_excel_data_semantically(df: pd.DataFrame, structure_info: Dict[str, Any], metadata: dict) -> List[Document]:
    """基于语义相关性智能分块Excel数据"""
    documents = []

    # 策略1：创建整体摘要文档
    summary_text = create_table_summary(df, metadata.get('sheet_name', ''), structure_info)
    summary_doc = Document(
        page_content=summary_text,
        metadata={
            **metadata,
            'content_type': 'table_summary',
            'chunk_strategy': 'summary',
            'file_type': 'excel'
        }
    )
    documents.append(summary_doc)

    # 策略2：创建列关系分析文档
    if len(df) > 5:  # 只有足够数据时才分析关系
        relationship_text = create_column_relationship_text(df, structure_info)
        if len(relationship_text) > 100:  # 确保有足够内容
            relationship_doc = Document(
                page_content=relationship_text,
                metadata={
                    **metadata,
                    'content_type': 'column_relationships',
                    'chunk_strategy': 'relationship_analysis',
                    'file_type': 'excel'
                }
            )
            documents.append(relationship_doc)

    # 策略3：按语义组分块数据行
    rows_per_chunk = calculate_optimal_chunk_size(structure_info)
    total_rows = len(df)

    for start_row in range(0, total_rows, rows_per_chunk):
        # 创建数据块
        data_text = format_table_data_intelligently(df, structure_info, start_row, rows_per_chunk)

        if is_valid_content(data_text, min_length=50):
            chunk_metadata = {
                **metadata,
                'content_type': 'table_data',
                'chunk_strategy': 'semantic_rows',
                'start_row': start_row + 1,
                'end_row': min(start_row + rows_per_chunk, total_rows),
                'chunk_size': min(rows_per_chunk, total_rows - start_row),
                'file_type': 'excel'
            }

            data_doc = Document(
                page_content=data_text,
                metadata=chunk_metadata
            )
            documents.append(data_doc)

    # 策略4：如果有多个重要的分类列，创建分类聚合文档
    for cat_col in structure_info.get('categorical_columns', [])[:2]:  # 最多处理2个分类列
        category_docs = create_category_aggregation_docs(df, cat_col, structure_info, metadata)
        documents.extend(category_docs)

    return documents


def calculate_optimal_chunk_size(structure_info: Dict[str, Any]) -> int:
    """根据表格结构计算最优的分块大小"""
    base_chunk_size = 15  # 基础行数

    # 根据列数调整
    col_count = structure_info['col_count']
    if col_count > 10:
        base_chunk_size = max(8, base_chunk_size - (col_count - 10) * 2)
    elif col_count < 5:
        base_chunk_size = min(25, base_chunk_size + (5 - col_count) * 3)

    # 根据数据密度调整
    density = structure_info['data_density']
    if density < 0.7:  # 数据稀疏时增加行数
        base_chunk_size = int(base_chunk_size * 1.5)
    elif density > 0.95:  # 数据密集时减少行数
        base_chunk_size = int(base_chunk_size * 0.8)

    return max(5, min(30, base_chunk_size))  # 限制在5-30行之间


def create_category_aggregation_docs(df: pd.DataFrame, cat_col: str, structure_info: Dict[str, Any], metadata: dict) -> \
List[Document]:
    """为重要分类列创建聚合文档"""
    documents = []

    # 获取分类值的分布
    category_counts = df[cat_col].value_counts()

    # 只处理出现频率较高的分类
    for category, count in category_counts.head(5).items():
        if pd.isna(category) or count < 2:
            continue

        # 获取该分类下的所有记录
        category_data = df[df[cat_col] == category]

        # 创建该分类的描述文档
        category_text_parts = [
            f"=== {cat_col}：{category} 的相关记录 ===",
            f"该分类包含{count}条记录，占总数据的{count / len(df):.1%}"
        ]

        # 添加该分类下其他列的特征描述
        for col in category_data.columns:
            if col == cat_col:
                continue

            col_type = structure_info['column_types'].get(col, 'text')
            if col_type == 'numeric':
                numeric_data = category_data[col].dropna()
                if len(numeric_data) > 0:
                    mean_val = numeric_data.mean()
                    category_text_parts.append(f"在{col}方面：平均值为{mean_val:.2f}")
            elif col_type in ['text', 'categorical']:
                text_data = category_data[col].dropna()
                if len(text_data) > 0:
                    most_common = text_data.mode()
                    if len(most_common) > 0:
                        category_text_parts.append(f"在{col}方面：最常见的是{most_common.iloc[0]}")

        # 添加几个具体的记录示例
        sample_records = []
        for idx, (_, row) in enumerate(category_data.head(3).iterrows()):
            record_parts = []
            for col, val in row.items():
                if col != cat_col and not pd.isna(val) and str(val).strip():
                    formatted_val = format_cell_value(val, structure_info['column_types'].get(col, 'text'))
                    record_parts.append(f"{col}为{formatted_val}")

            if record_parts:
                sample_records.append(f"示例{idx + 1}：" + "，".join(record_parts[:4]))

        if sample_records:
            category_text_parts.append("具体记录示例：")
            category_text_parts.extend(sample_records)

        category_text = "\n".join(category_text_parts)

        if is_valid_content(category_text, min_length=80):
            category_doc = Document(
                page_content=category_text,
                metadata={
                    **metadata,
                    'content_type': 'category_aggregation',
                    'chunk_strategy': 'category_based',
                    'category_column': cat_col,
                    'category_value': str(category),
                    'record_count': count,
                    'file_type': 'excel'
                }
            )
            documents.append(category_doc)

    return documents


def load_excel_with_enhanced_strategies(excel_path: str) -> List[Document]:
    """使用增强策略加载Excel文件，优化embedding效果"""
    documents = []

    try:
        print(f"  正在使用增强策略加载Excel文件: {os.path.basename(excel_path)}")

        # 读取Excel文件，获取所有工作表
        excel_file = pd.ExcelFile(excel_path)
        print(f"  发现 {len(excel_file.sheet_names)} 个工作表: {excel_file.sheet_names}")

        for sheet_name in excel_file.sheet_names:
            try:
                print(f"    处理工作表: {sheet_name}")

                # 读取工作表数据，保持数据类型
                df = pd.read_excel(excel_path, sheet_name=sheet_name)

                if df.empty:
                    print(f"    工作表 {sheet_name} 为空，跳过")
                    continue

                # 清理列名
                df.columns = [str(col).strip() for col in df.columns]

                # 分析表格结构
                structure_info = analyze_excel_structure(df)
                print(f"    表格结构分析：{structure_info['row_count']}行×{structure_info['col_count']}列，"
                      f"数据密度{structure_info['data_density']:.1%}")
                print(f"    列类型分布：数值{len(structure_info['numeric_columns'])}个，"
                      f"文本{len(structure_info['text_columns'])}个，"
                      f"分类{len(structure_info['categorical_columns'])}个")

                # 基础元数据（只包含简单数据类型）
                base_metadata = {
                    "source": excel_path,
                    "source_file": os.path.basename(excel_path),
                    "sheet_name": sheet_name,
                    "extraction_method": "enhanced_pandas_excel",
                    "rows_count": structure_info['row_count'],
                    "columns_count": structure_info['col_count'],
                    "data_density": structure_info['data_density'],
                    "numeric_columns_count": len(structure_info['numeric_columns']),
                    "text_columns_count": len(structure_info['text_columns']),
                    "categorical_columns_count": len(structure_info['categorical_columns']),
                    "date_columns_count": len(structure_info['date_columns']),
                    "file_type": "excel"
                }

                # 使用语义分块策略
                sheet_documents = split_excel_data_semantically(df, structure_info, base_metadata)

                print(f"    工作表 {sheet_name} 生成了 {len(sheet_documents)} 个语义文档块")
                documents.extend(sheet_documents)

            except Exception as e:
                print(f"    处理工作表 {sheet_name} 时出错: {e}")
                continue

        if documents:
            print(f"  成功从Excel文件提取了 {len(documents)} 个增强型文档块")
        else:
            print(f"  Excel文件未提取到有效内容")

    except Exception as e:
        print(f"  加载Excel文件失败: {e}")

    return documents


# =============================================================================
# 统一的文档处理和向量化主函数
# =============================================================================

def process_all_documents():
    """QA增强的文档处理主函数"""

    print("🚀 开始QA增强的文档处理流程")
    print("=" * 80)

    # 初始化embeddings和向量数据库
    print("🌐 正在初始化阿里云DashScope Embedding API连接...")
    
    # 使用阿里云的text-embedding-v4模型（自定义直连DashScope以避免兼容层格式问题）
    embeddings = DashScopeEmbeddings(
        api_key=dashscope_api_key,
        model="text-embedding-v4",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        request_timeout_seconds=30,
        sleep_between_requests_seconds=0.05,
    )
    print("✅ 成功初始化阿里云text-embedding-v4（直连DashScope兼容端点）")

    # 创建向量数据库
    vector_store = Chroma(
        collection_name="aliyun_text_embedding_v4_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    print(f"📦 向量数据库初始化完成：aliyun_text_embedding_v4_collection")
    print(f"💾 存储位置：./chroma_langchain_db")
    print(f"🎯 使用模型：text-embedding-v4")

    all_documents = []
    all_qa_documents = []

    # =============================================================================
    # 步骤1：创建预定义目标QA文档
    # =============================================================================

    print("\n" + "=" * 60)
    print("📋 步骤1：创建预定义目标QA文档集合")
    print("=" * 60)

    target_qa_docs = create_target_qa_documents()
    all_qa_documents.extend(target_qa_docs)

    # =============================================================================
    # 步骤2：处理PDF文件并生成QA
    # =============================================================================

    print("\n" + "=" * 60)
    print("📄 步骤2：处理PDF文件并生成QA对")
    print("=" * 60)

    pdf_folder = "./datas/"
    if os.path.exists(pdf_folder):
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        print(f"🔍 找到 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")

        pdf_documents = []

        # 处理每个PDF文件
        for pdf_index, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 正在处理第 {pdf_index}/{len(pdf_files)} 个PDF文件: {os.path.basename(pdf_file)}")

            # 使用改进的加载策略
            documents = load_pdf_with_multiple_strategies(pdf_file)

            if not documents:
                print(f"  跳过PDF文件 {os.path.basename(pdf_file)}：无法提取有效内容")
                continue

            # 使用章节切分策略
            for doc in documents:
                splits = split_by_chapters(doc.page_content, {
                    'source': pdf_file,
                    'source_file': os.path.basename(pdf_file),
                    'extraction_method': doc.metadata.get('extraction_method', 'unknown'),
                    'file_type': 'pdf'
                })

                print(f"  按章节分割成 {len(splits)} 个文档块")

                # 过滤和清理文档
                valid_splits = []
                for split_doc in splits:
                    # 首先清理内容
                    cleaned_content = clean_text(split_doc.page_content)
                    split_doc.page_content = cleaned_content

                    # 然后验证是否有效
                    if is_valid_content(cleaned_content) and validate_document_before_embedding(split_doc):
                        split_doc.metadata.update({
                            'content_length': len(cleaned_content),
                        })
                        valid_splits.append(split_doc)

                print(f"  过滤后获得 {len(valid_splits)} 个高质量PDF章节块")
                all_documents.extend(valid_splits)

                # 为每个有效的文档块生成QA对
                print(f"  开始为PDF章节生成QA对...")
                pdf_qa_docs = []

                for i, split_doc in enumerate(valid_splits[:3]):  # 限制前3个文档块生成QA
                    print(f"    为第 {i + 1}/{min(3, len(valid_splits))} 个文档块生成QA对")
                    qa_docs = generate_qa_pairs_from_document(split_doc)
                    pdf_qa_docs.extend(qa_docs)

                    # 添加延迟避免API限制
                    if qa_docs:
                        time.sleep(1)

                print(f"  PDF文件共生成 {len(pdf_qa_docs)} 个QA文档")
                all_qa_documents.extend(pdf_qa_docs)

        print(
            f"\n📊 PDF处理完成，共获得 {len([d for d in all_documents if d.metadata.get('file_type') == 'pdf'])} 个PDF文档块")
    else:
        print(f"⚠️ PDF文件夹 {pdf_folder} 不存在，跳过PDF处理")

    # =============================================================================
    # 步骤3：处理Excel文件并生成QA
    # =============================================================================

    print("\n" + "=" * 60)
    print("📊 步骤3：处理Excel文件并生成QA对")
    print("=" * 60)

    excel_folder = "./datas/"
    if os.path.exists(excel_folder):
        excel_files = []
        for ext in ['*.xlsx', '*.xls']:
            excel_files.extend(glob.glob(os.path.join(excel_folder, ext)))

        print(f"\n📊 找到 {len(excel_files)} 个Excel文件:")
        for excel_file in excel_files:
            print(f"  - {excel_file}")

        excel_documents = []

        # 使用增强策略处理每个Excel文件
        for excel_index, excel_file in enumerate(excel_files, 1):
            print(f"\n📈 正在处理第 {excel_index}/{len(excel_files)} 个Excel文件: {os.path.basename(excel_file)}")

            # 使用增强的Excel加载策略
            documents = load_excel_with_enhanced_strategies(excel_file)

            if not documents:
                print(f"  跳过Excel文件 {os.path.basename(excel_file)}：无法提取有效内容")
                continue

            # 过滤和验证文档
            valid_documents = []
            for doc in documents:
                # 清理内容
                cleaned_content = clean_text(doc.page_content)
                doc.page_content = cleaned_content

                # 验证内容质量
                if is_valid_content(cleaned_content) and validate_document_before_embedding(doc, min_length=50):
                    doc.metadata.update({
                        'content_length': len(cleaned_content),
                        'processing_version': 'enhanced_v2'
                    })
                    valid_documents.append(doc)

            print(f"  文件处理完成，获得 {len(valid_documents)} 个高质量Excel文档块")
            all_documents.extend(valid_documents)

            # 为Excel文档生成QA对
            print(f"  开始为Excel文档生成QA对...")
            excel_qa_docs = []

            for i, doc in enumerate(valid_documents):
                print(f"    为第 {i + 1}/{len(valid_documents)} 个Excel文档生成QA对")
                qa_docs = generate_qa_pairs_from_document(doc)
                excel_qa_docs.extend(qa_docs)

                # 添加延迟避免API限制
                if qa_docs:
                    time.sleep(1)

            print(f"  Excel文件共生成 {len(excel_qa_docs)} 个QA文档")
            all_qa_documents.extend(excel_qa_docs)

        print(
            f"\n📊 Excel处理完成，共获得 {len([d for d in all_documents if d.metadata.get('file_type') == 'excel'])} 个Excel文档块")
    else:
        print(f"⚠️ Excel文件夹 {excel_folder} 不存在，跳过Excel处理")

    # =============================================================================
    # 步骤4：统一入库到向量数据库
    # =============================================================================

    print("\n" + "=" * 60)
    print("🎯 步骤4：统一入库到向量数据库")
    print("=" * 60)

    # 合并所有文档
    all_final_documents = all_documents + all_qa_documents

    print(f"\n📊 文档统计信息：")
    print(f"  - 原始文档块: {len(all_documents)} 个")
    print(f"  - QA文档块: {len(all_qa_documents)} 个")
    print(f"    - 预定义目标QA: {len(target_qa_docs)} 个")
    print(f"    - 生成的QA: {len(all_qa_documents) - len(target_qa_docs)} 个")
    print(f"  - 文档总计: {len(all_final_documents)} 个")

    if len(all_final_documents) > 0:
        # 显示处理样本
        print(f"\n=== 📋 QA增强文档处理样本预览 ===")

        # 按文档类型分组显示样本
        doc_types = {}
        for doc in all_final_documents:
            doc_type = doc.metadata.get('content_type', doc.metadata.get('file_type', 'unknown'))
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)

        for doc_type, docs in list(doc_types.items())[:5]:  # 显示前5种类型
            print(f"\n--- {doc_type} 类型样本 ---")
            sample_doc = docs[0]
            print(f"来源: {sample_doc.metadata.get('source_file', 'unknown')}")
            print(f"内容类型: {sample_doc.metadata.get('content_type', 'unknown')}")

            if sample_doc.metadata.get('content_type') in ['generated_qa', 'target_qa']:
                print(f"问题: {sample_doc.metadata.get('question', '未知')[:50]}...")
                print(f"生成方法: {sample_doc.metadata.get('generation_method', 'unknown')}")
            else:
                print(f"文件类型: {sample_doc.metadata.get('file_type', 'unknown')}")

            print(f"长度: {sample_doc.metadata.get('content_length', len(sample_doc.page_content))}")
            print(f"内容预览: {sample_doc.page_content[:150]}...")
            print("-" * 50)

        # 分批添加到向量数据库
        batch_size = 5
        total_batches = (len(all_final_documents) + batch_size - 1) // batch_size
        successful_docs = 0

        for i in range(0, len(all_final_documents), batch_size):
            batch = all_final_documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\n🔄 正在处理第 {batch_num}/{total_batches} 批文档，包含 {len(batch)} 个文档")

            # 最终验证
            validated_batch = []
            for j, doc in enumerate(batch):
                content_length = len(doc.page_content.strip())
                content_type = doc.metadata.get('content_type', doc.metadata.get('file_type', 'unknown'))
                print(
                    f"    文档 {j + 1}: {content_type}, 长度={content_length}, 来源={doc.metadata.get('source_file', 'unknown')}")

                min_length = 30 if content_type in ['generated_qa', 'target_qa'] else 50
                if validate_document_before_embedding(doc, min_length=min_length):
                    validated_batch.append(doc)
                else:
                    print(f"    ⚠️ 跳过无效文档 {j + 1}")

            if not validated_batch:
                print(f"    批次 {batch_num} 没有有效文档，跳过")
                continue

            print(f"    验证后批次包含 {len(validated_batch)} 个有效文档")

            try:
                # 确保所有文档内容都是字符串格式
                cleaned_batch = []
                for doc in validated_batch:
                    # 确保page_content是字符串
                    if not isinstance(doc.page_content, str):
                        doc.page_content = str(doc.page_content)
                    # 清理内容，移除可能的特殊字符
                    doc.page_content = doc.page_content.strip()
                    if doc.page_content:
                        cleaned_batch.append(doc)
                
                if cleaned_batch:
                    _ = vector_store.add_documents(documents=cleaned_batch)
                    successful_docs += len(cleaned_batch)
                    print(f"✅ 成功添加第 {batch_num} 批文档")
                else:
                    print(f"⚠️ 批次 {batch_num} 清理后没有有效文档")
            except Exception as e:
                print(f"❌ 添加第 {batch_num} 批文档时出错: {e}")
                # 逐个尝试添加
                for k, doc in enumerate(validated_batch):
                    try:
                        # 确保单个文档内容格式正确
                        if not isinstance(doc.page_content, str):
                            doc.page_content = str(doc.page_content)
                        doc.page_content = doc.page_content.strip()
                        
                        if doc.page_content:
                            vector_store.add_documents(documents=[doc])
                            successful_docs += 1
                            print(f"      ✅ 成功添加单个文档 {k + 1}")
                        else:
                            print(f"      ⚠️ 文档 {k + 1} 内容为空，跳过")
                    except Exception as single_e:
                        print(f"      ❌ 文档 {k + 1} 添加失败: {single_e}")

        print(f"\n🎉 QA增强处理完成！")
        print(f"📊 最终统计信息：")
        pdf_count = len(glob.glob(os.path.join('./datas/', '*.pdf'))) if os.path.exists('./datas/') else 0
        excel_count = len([f for ext in ['*.xlsx', '*.xls'] for f in
                           glob.glob(os.path.join('./datas/', ext))]) if os.path.exists('./datas/') else 0
        print(f"  - 处理的PDF文件数: {pdf_count}")
        print(f"  - 处理的Excel文件数: {excel_count}")
        print(f"  - 原始文档块总数: {len(all_documents)}")
        print(f"  - QA文档块总数: {len(all_qa_documents)}")
        print(f"  - 生成的文档块总数: {len(all_final_documents)}")
        print(f"  - 成功添加到向量库: {successful_docs}")
        print(f"🔍 使用了QA数据增强策略：原始文档 + 生成QA + 目标QA")
        print(f"💾 向量库保存位置: ./chroma_langchain_db")
        print(f"📈 向量库集合名称: aliyun_text_embedding_v4_collection")
        print(f"🎯 使用模型: text-embedding-v4")
        print(f"🌐 运行模式: 阿里云DashScope API")
        print(f"🏆 模型性能: 阿里云text-embedding-v4，高质量向量表示")
        
        print(f"\n🎉 基于阿里云text-embedding-v4的QA增强处理完成！")
        print(f"🎯 使用模型：text-embedding-v4")
        print(f"🌐 服务模式：阿里云DashScope API")
        print(f"🏆 模型性能：阿里云text-embedding-v4，高质量向量表示")
        print(f"📏 技术规格：高维度向量嵌入，多语言支持")
        print(f"💪 优势特色：稳定云端服务，高精度向量表示，医学领域优化")
        
    else:
        print("❌ 没有提取到有效的文档内容，请检查文件路径和文件格式")




# =============================================================================
# 执行主函数
# =============================================================================

if __name__ == "__main__":
    # 运行文档处理流程
    process_all_documents() 