import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import hashlib
from datetime import datetime, timedelta
import pickle
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalAttachmentProcessor")

DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")

# 医疗附件缓存管理器
class MedicalAttachmentCache:
    def __init__(self, cache_dir: str = "medical_cache", max_cache_size: int = 100, cache_expiry_hours: int = 24):
        """
        医疗附件分析结果缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            max_cache_size: 最大缓存条目数量
            cache_expiry_hours: 缓存过期时间（小时）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # 启动时清理过期缓存
        self._cleanup_expired_cache()
        
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存索引"""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载缓存索引失败: {e}")
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _generate_cache_key(self, user_ip: str, filename: str, file_size: int = None) -> str:
        """
        生成缓存键值
        
        Args:
            user_ip: 用户IP
            filename: 文件名
            file_size: 文件大小（可选，用于更精确的缓存键）
            
        Returns:
            缓存键值的MD5哈希
        """
        # 使用用户IP、文件名和文件大小生成唯一键值
        cache_string = f"{user_ip}_{filename}"
        if file_size:
            cache_string += f"_{file_size}"
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def _cleanup_expired_cache(self):
        """清理过期的缓存条目"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, cache_info in self.cache_index.items():
            cache_time = datetime.fromisoformat(cache_info['created_at'])
            if current_time - cache_time > timedelta(hours=self.cache_expiry_hours):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _remove_cache_entry(self, cache_key: str):
        """删除缓存条目"""
        try:
            if cache_key in self.cache_index:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[cache_key]
                logger.debug(f"删除缓存条目: {cache_key}")
        except Exception as e:
            logger.warning(f"删除缓存条目失败 {cache_key}: {e}")
    
    def _enforce_cache_size_limit(self):
        """强制执行缓存大小限制"""
        if len(self.cache_index) <= self.max_cache_size:
            return
        
        # 按创建时间排序，删除最旧的条目
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]['created_at']
        )
        
        entries_to_remove = len(self.cache_index) - self.max_cache_size
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            self._remove_cache_entry(cache_key)
        
        logger.info(f"强制清理了 {entries_to_remove} 个旧缓存条目以控制缓存大小")
    
    def get_cached_analysis(self, user_ip: str, filename: str, file_size: int = None) -> Optional[Dict[str, Any]]:
        """
        获取缓存的分析结果
        
        Args:
            user_ip: 用户IP
            filename: 文件名
            file_size: 文件大小
            
        Returns:
            缓存的分析结果或None
        """
        cache_key = self._generate_cache_key(user_ip, filename, file_size)
        
        if cache_key not in self.cache_index:
            return None
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if not cache_file.exists():
                # 索引存在但文件不存在，清理无效索引
                del self.cache_index[cache_key]
                self._save_cache_index()
                return None
            
            # 检查是否过期
            cache_info = self.cache_index[cache_key]
            cache_time = datetime.fromisoformat(cache_info['created_at'])
            if datetime.now() - cache_time > timedelta(hours=self.cache_expiry_hours):
                self._remove_cache_entry(cache_key)
                self._save_cache_index()
                return None
            
            # 加载缓存数据
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 更新访问时间
            self.cache_index[cache_key]['last_accessed'] = datetime.now().isoformat()
            self._save_cache_index()
            
            logger.info(f"✅ 命中医疗附件缓存: {filename} (用户: {user_ip})")
            return cached_data
            
        except Exception as e:
            logger.warning(f"读取缓存失败 {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            self._save_cache_index()
            return None
    
    def cache_analysis_result(self, user_ip: str, filename: str, analysis_result: Dict[str, Any], file_size: int = None):
        """
        缓存分析结果
        
        Args:
            user_ip: 用户IP
            filename: 文件名
            analysis_result: 分析结果
            file_size: 文件大小
        """
        try:
            cache_key = self._generate_cache_key(user_ip, filename, file_size)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # 保存分析结果到文件
            with open(cache_file, 'wb') as f:
                pickle.dump(analysis_result, f)
            
            # 更新索引
            self.cache_index[cache_key] = {
                'user_ip': user_ip,
                'filename': filename,
                'file_size': file_size,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'cache_file': f"{cache_key}.pkl"
            }
            
            # 强制执行缓存大小限制
            self._enforce_cache_size_limit()
            
            # 保存索引
            self._save_cache_index()
            
            logger.info(f"📁 医疗附件分析结果已缓存: {filename} (用户: {user_ip})")
            
        except Exception as e:
            logger.error(f"缓存分析结果失败: {e}")
    
    def clear_user_cache(self, user_ip: str):
        """清理特定用户的所有缓存"""
        removed_count = 0
        keys_to_remove = []
        
        for cache_key, cache_info in self.cache_index.items():
            if cache_info.get('user_ip') == user_ip:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            self._remove_cache_entry(key)
            removed_count += 1
        
        self._save_cache_index()
        logger.info(f"清理用户 {user_ip} 的 {removed_count} 个缓存条目")
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_entries = len(self.cache_index)
        total_size = sum(
            (self.cache_dir / f"{key}.pkl").stat().st_size 
            for key in self.cache_index.keys()
            if (self.cache_dir / f"{key}.pkl").exists()
        )
        
        return {
            'total_entries': total_entries,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir),
            'max_cache_size': self.max_cache_size,
            'cache_expiry_hours': self.cache_expiry_hours
        }

# S3客户端配置
class S3Config:
    def __init__(self):
        # 从环境变量读取S3配置，如果没有设置则使用默认值
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://154.89.148.156:9000")
        self.access_key = os.getenv("S3_ACCESS_KEY", )
        self.secret_key = os.getenv("S3_SECRET_KEY",)
        self.region_name = os.getenv("S3_REGION", "us-east-1")  # MinIO通常使用这个默认值
        
    def get_s3_client(self):
        """获取配置好的S3客户端"""
        return boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name
        )

# 医疗附件处理器
class MedicalAttachmentProcessor:
    def __init__(self):
        # 初始化阿里云DashScope qwen-vl-max模型用于医疗图像分析（添加超时和重试配置）
        self.vision_model = ChatOpenAI(
            model="qwen-vl-max",  # 阿里云视觉理解模型
            temperature=0.1,
            request_timeout=90,  # 图像分析需要更长超时时间（90秒）
            max_retries=2,       # 最大重试2次
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        # 创建上传目录
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # 支持的文件类型
        self.supported_image_types = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.supported_doc_types = {'.pdf', '.txt', '.docx'}
        
        # 初始化S3配置
        self.s3_config = S3Config()
        
        # 初始化缓存管理器
        self.cache_manager = MedicalAttachmentCache(
            cache_dir="medical_cache",
            max_cache_size=100,  # 最多缓存100个文件的分析结果
            cache_expiry_hours=24  # 缓存24小时后过期
        )
        
        logger.info(f"医疗附件处理器初始化完成，缓存配置: {self.cache_manager.get_cache_stats()}")
        
    async def download_from_s3(self, bucket_name: str, file_key: str) -> Tuple[bytes, str]:
        """
        从S3下载文件
        
        Args:
            bucket_name: S3桶名
            file_key: 文件的S3键名
            
        Returns:
            文件内容字节和文件名的元组
        """
        try:
            s3_client = self.s3_config.get_s3_client()
            
            # 下载文件到内存
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response['Body'].read()
            
            # 从file_key中提取文件名
            filename = os.path.basename(file_key)
            
            logger.info(f"成功从S3下载文件: {bucket_name}/{file_key}")
            return file_content, filename
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"S3文件不存在: {bucket_name}/{file_key}")
                raise FileNotFoundError(f"S3文件不存在: {bucket_name}/{file_key}")
            elif error_code == 'NoSuchBucket':
                logger.error(f"S3桶不存在: {bucket_name}")
                raise FileNotFoundError(f"S3桶不存在: {bucket_name}")
            else:
                logger.error(f"下载S3文件时出错: {e}")
                raise Exception(f"下载S3文件时出错: {e}")
        except NoCredentialsError:
            logger.error("S3凭证配置错误")
            raise Exception("S3凭证配置错误")
        except Exception as e:
            logger.error(f"下载S3文件时发生未知错误: {e}")
            raise Exception(f"下载S3文件时发生未知错误: {e}")

    async def process_medical_attachment_from_s3(self, user_ip: str, filename: str) -> Dict[str, Any]:
        """
        从S3处理医疗附件的主函数（支持缓存）
        
        Args:
            user_ip: 用户IP，用作S3桶名的一部分
            filename: 文件名
            
        Returns:
            处理结果字典
        """
        try:
            # 1. 首先检查缓存（增加快速路径）
            logger.info(f"🔍 检查医疗附件缓存: {filename} (用户: {user_ip})")
            cached_result = self.cache_manager.get_cached_analysis(user_ip, filename)
            
            if cached_result:
                logger.info(f"⚡ 缓存命中，直接返回: {filename}")
                # 返回缓存的结果，但更新一些元数据
                cached_result["status"] = "success_from_cache"
                cached_result["original_filename"] = filename
                return cached_result
            
            # 2. 缓存未命中，进行完整处理
            logger.info(f"💾 缓存未命中，开始处理医疗附件: {filename}")
            
            # 生成符合S3命名规范的桶名
            safe_ip = user_ip.replace(".", "-").replace(":", "-").lower()
            bucket_prefix = os.getenv("S3_USER_BUCKET_PREFIX", "user")
            bucket_name = f"{bucket_prefix}"
            
            logger.info(f"尝试从S3桶 {bucket_name} 下载文件: {filename}")
            
            # 从S3下载文件
            file_content, actual_filename = await self.download_from_s3(bucket_name, filename)
            file_size = len(file_content)
            
            # 使用现有的处理逻辑
            analysis_result = await self.process_medical_attachment(
                file_content=file_content,
                filename=actual_filename,
                file_type="s3_medical_attachment",
                user_ip=user_ip
            )
            
            # 3. 如果处理成功，缓存结果
            if analysis_result.get("status") == "success":
                logger.info(f"💾 缓存医疗附件分析结果: {filename}")
                self.cache_manager.cache_analysis_result(
                    user_ip=user_ip,
                    filename=filename,
                    analysis_result=analysis_result,
                    file_size=file_size
                )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"从S3处理医疗附件时出错: {e}")
            return {
                "file_id": "",
                "file_path": "",
                "extracted_content": f"从S3处理文件时出错: {str(e)}",
                "file_type": "s3_medical_attachment",
                "original_filename": filename,
                "status": "error"
            }

    async def test_s3_connection(self) -> bool:
        """
        测试S3连接是否正常
        
        Returns:
            连接是否成功
        """
        try:
            s3_client = self.s3_config.get_s3_client()
            # 尝试列出桶来测试连接
            response = s3_client.list_buckets()
            logger.info("S3连接测试成功")
            return True
        except Exception as e:
            logger.error(f"S3连接测试失败: {e}")
            return False
    
    def cleanup_temp_file(self, file_path: str):
        """
        清理临时文件
        
        Args:
            file_path: 要清理的文件路径
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"已清理临时文件: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    async def _compress_image(self, file_path: Path, max_size: tuple = (1600, 1600), quality: int = 85, target_base64_size: int = None, is_medical: bool = True) -> bytes:
        """
        压缩图像以减少文件大小，避免API字符限制
        
        Args:
            file_path: 图像文件路径
            max_size: 最大尺寸 (width, height) - 医疗图像增大默认尺寸
            quality: JPEG质量 (1-100) - 医疗图像提高默认质量
            target_base64_size: 目标base64编码大小（字节）
            is_medical: 是否为医疗图像，启用特殊优化
            
        Returns:
            压缩后的图像数据
        """
        try:
            with Image.open(file_path) as img:
                # 转换为RGB模式（确保兼容性）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 医疗图像预处理 - 增强对比度和锐化
                if is_medical:
                    img = self._enhance_medical_image(img)
                
                # 如果指定了目标base64大小，使用自适应压缩
                if target_base64_size:
                    return await self._adaptive_compress(img, target_base64_size, is_medical)
                
                # 计算缩放比例
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 保存到字节流
                import io
                output = io.BytesIO()
                
                # 统一转为JPEG格式以获得最佳压缩率
                img.save(output, format='JPEG', quality=quality, optimize=True)
                
                compressed_data = output.getvalue()
                original_size = os.path.getsize(file_path)
                compressed_size = len(compressed_data)
                
                logger.info(f"图像压缩完成: {original_size} bytes → {compressed_size} bytes "
                          f"(压缩率: {(1 - compressed_size/original_size)*100:.1f}%)")
                
                return compressed_data
                
        except Exception as e:
            logger.error(f"图像压缩失败: {e}")
            # 如果压缩失败，返回原始数据
            with open(file_path, 'rb') as f:
                return f.read()
    
    async def _adaptive_compress(self, img: Image.Image, target_base64_size: int, is_medical: bool = True) -> bytes:
        """
        自适应压缩图像到指定的base64大小
        
        Args:
            img: PIL图像对象
            target_base64_size: 目标base64编码大小（字节）
            is_medical: 是否为医疗图像，影响压缩策略
            
        Returns:
            压缩后的图像数据
        """
        import io
        
        # 由于base64编码会增加约33%的大小，计算目标原始数据大小
        target_raw_size = int(target_base64_size * 0.75)
        
        # 为医疗图像设置更保守的初始参数
        if is_medical:
            max_width = 1800  # 医疗图像保持更高分辨率
            quality = 95      # 医疗图像保持更高质量
            min_quality = 70  # 医疗图像最低质量进一步提高
            min_width = 800   # 医疗图像最小宽度提高
        else:
            max_width = 1000  # 非医疗图像也适当提高
            quality = 85
            min_quality = 20  # 非医疗图像最低质量也适当提高
            min_width = 200
        
        for attempt in range(10):  # 最多尝试10次
            # 创建图像副本用于压缩
            img_copy = img.copy()
            
            # 按比例缩放
            if max_width < img_copy.width:
                ratio = max_width / img_copy.width
                new_height = int(img_copy.height * ratio)
                img_copy = img_copy.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # 压缩为JPEG
            output = io.BytesIO()
            img_copy.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_data = output.getvalue()
            
            logger.info(f"自适应压缩尝试 {attempt + 1}: 尺寸={img_copy.width}x{img_copy.height}, "
                       f"质量={quality}, 大小={len(compressed_data)} bytes")
            
            # 检查是否满足目标大小
            if len(compressed_data) <= target_raw_size:
                return compressed_data
            
            # 调整压缩参数 - 医疗图像优先保持质量
            if is_medical:
                if len(compressed_data) > target_raw_size * 3:
                    # 大幅超出，优先缩小尺寸，少降质量
                    max_width = int(max_width * 0.85)
                    quality = max(min_quality, quality - 3)
                elif len(compressed_data) > target_raw_size * 2:
                    # 中等超出，平衡调整，但更保守
                    max_width = int(max_width * 0.9)
                    quality = max(min_quality, quality - 5)
                else:
                    # 略微超出，轻微调整质量
                    quality = max(min_quality, quality - 8)
            else:
                # 非医疗图像使用原有策略
                if len(compressed_data) > target_raw_size * 2:
                    max_width = int(max_width * 0.7)
                    quality = max(min_quality, quality - 10)
                elif len(compressed_data) > target_raw_size * 1.5:
                    max_width = int(max_width * 0.8)
                    quality = max(min_quality, quality - 15)
                else:
                    quality = max(min_quality, quality - 20)
            
            # 防止无限循环
            if max_width < min_width or quality < min_quality:
                break
        
        # 如果仍然过大，使用保守的最小设置
        logger.warning("自适应压缩未能达到目标大小，使用医疗图像保守设置")
        img_copy = img.copy()
        
        if is_medical:
            # 医疗图像保持更大的最小尺寸和质量 - 提高最小标准
            img_copy.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            final_quality = 75  # 提高最终质量
        else:
            img_copy.thumbnail((400, 400), Image.Resampling.LANCZOS)  # 非医疗图像也适当提高
            final_quality = 30
            
        output = io.BytesIO()
        img_copy.save(output, format='JPEG', quality=final_quality, optimize=True)
        
        return output.getvalue()

    def _enhance_medical_image(self, img: Image.Image) -> Image.Image:
        """
        医疗图像增强处理，提高对比度和细节清晰度
        
        Args:
            img: PIL图像对象
            
        Returns:
            增强后的图像
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # 1. 对比度增强 - 对医疗影像特别重要
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.3)  # 适度增强对比度
            
            # 2. 锐化处理 - 增强边缘和细节
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.2)  # 适度锐化
            
            # 3. 亮度调整 - 确保细节可见
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.1)  # 轻微增亮
            
            # 4. 应用轻微的锐化滤镜
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=20, threshold=3))
            
            logger.info("✅ 医疗图像增强处理完成")
            return img
            
        except Exception as e:
            logger.warning(f"医疗图像增强失败，使用原图: {e}")
            return img

    async def process_medical_attachment(self, file_content: bytes, filename: str, 
                                       file_type: str, user_ip: str) -> Dict[str, Any]:
        """
        处理医疗附件的主函数
        
        Args:
            file_content: 文件二进制内容
            filename: 文件名
            file_type: 文件类型标识
            user_ip: 用户IP
            
        Returns:
            处理结果字典
        """
        temp_file_path = None
        try:
            # 生成唯一的文件ID
            file_id = str(uuid.uuid4())
            file_extension = Path(filename).suffix.lower()
            
            # 保存文件到临时目录
            safe_filename = f"{file_id}_{filename}"
            file_path = self.upload_dir / safe_filename
            temp_file_path = str(file_path)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"文件已保存: {file_path}")
            
            # 根据文件类型处理
            if file_extension in self.supported_image_types:
                extracted_content = await self._process_medical_image(file_path)
            elif file_extension == '.pdf':
                extracted_content = await self._process_medical_pdf(file_path)
            elif file_extension in {'.txt', '.docx'}:
                extracted_content = await self._process_medical_document(file_path)
            else:
                extracted_content = f"不支持的文件类型: {file_extension}"
            
            result = {
                "file_id": file_id,
                "file_path": str(file_path),
                "extracted_content": extracted_content,
                "file_type": file_type,
                "original_filename": filename,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"处理医疗附件时出错: {e}")
            return {
                "file_id": "",
                "file_path": "",
                "extracted_content": f"处理文件时出错: {str(e)}",
                "file_type": file_type,
                "original_filename": filename,
                "status": "error"
            }
        finally:
            # 清理临时文件
            if temp_file_path:
                self.cleanup_temp_file(temp_file_path)
    
    async def process_medical_image_base64(self, base64_image_data: str, filename: str = "image.jpg", user_ip: str = "unknown") -> Dict[str, Any]:
        """直接处理Base64编码的医疗图像。
        
        支持带有或不带有 data URL 前缀的Base64字符串，例如：
        - data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
        - /9j/4AAQSkZJRgABAQ...
        """
        try:
            # 去除 data URL 前缀（如果存在）
            if base64_image_data.startswith("data:"):
                try:
                    base64_image_data = base64_image_data.split(",", 1)[1]
                except Exception:
                    pass
            # 解码为二进制
            image_bytes = base64.b64decode(base64_image_data)
            # 复用现有入口，统一处理与缓存逻辑
            return await self.process_medical_attachment(
                file_content=image_bytes,
                filename=filename,
                file_type="base64_medical_image",
                user_ip=user_ip,
            )
        except Exception as e:
            logger.error(f"处理Base64医疗图像失败: {e}")
            return {
                "file_id": "",
                "file_path": "",
                "extracted_content": f"处理Base64图像时出错: {str(e)}",
                "file_type": "base64_medical_image",
                "original_filename": filename,
                "status": "error",
            }
    
    async def _process_medical_image(self, file_path: Path) -> str:
        """处理医疗图像（X光片、CT、MRI、血检报告等）"""
        try:
            # 医疗图像使用更大的目标尺寸，确保关键信息不丢失
            target_base64_size = 40000  # 大幅提高目标大小，约30KB原始数据，确保医疗图像质量
            compressed_image_data = await self._compress_image(
                file_path, 
                target_base64_size=target_base64_size,
                is_medical=True  # 启用医疗图像特殊处理
            )
            base64_image = base64.b64encode(compressed_image_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_image}"

            # 构建医疗图像分析提示词（文本+图像数据URL）
            text_prompt = (
                "分析医疗图像，提供：\n"
                "1. 检查类型（X光/CT/MRI/血检等）\n"
                "2. 关键信息和异常发现\n"
                "3. 专业建议"
            )

            # 使用阿里云DashScope的OpenAI兼容多模态格式：text + image_url(data URL)
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]
            
            response = await self.vision_model.ainvoke(messages)
            
            # 限制图像分析结果长度，为下游RAG整合预留Token空间
            analysis_result = response.content
            max_analysis_length = 2000  # 增加分析结果长度限制
            
            if len(analysis_result) > max_analysis_length:
                # 智能截断：保留前部分关键信息
                truncated_result = analysis_result[:max_analysis_length]
                # 确保在句子边界截断
                last_period = truncated_result.rfind('。')
                last_newline = truncated_result.rfind('\n')
                if last_period > -1 and last_period > last_newline:
                    truncated_result = truncated_result[:last_period + 1]
                elif last_newline > -1:
                    truncated_result = truncated_result[:last_newline]
                
                truncated_result += "\n\n[注：分析结果已截断以确保系统稳定性]"
                logger.info(f"图像分析结果已截断：{len(analysis_result)} → {len(truncated_result)} 字符")
                return truncated_result
            
            logger.info(f"✅ 图像分析成功完成，结果长度: {len(analysis_result)} 字符")
            return analysis_result
            
        except Exception as e:
            logger.error(f"处理医疗图像时出错: {e}")
            # 如果AI分析失败，尝试OCR文本提取作为备用方案
            logger.info("尝试OCR文本提取作为备用方案...")
            try:
                ocr_result = await self._extract_text_from_image(file_path)
                return f"图像AI分析失败，OCR文本提取结果：\n{ocr_result}"
            except Exception as ocr_e:
                logger.error(f"OCR文本提取也失败: {ocr_e}")
                return f"图像分析失败: {str(e)}，OCR提取也失败: {str(ocr_e)}"
    
    async def _extract_text_from_image(self, file_path: Path) -> str:
        """从图像中提取文本作为备用方案（OCR）"""
        try:
            # 尝试导入OCR库
            try:
                import pytesseract  # type: ignore[import]
                # PIL Image已在文件顶部导入
            except ImportError:
                logger.warning("OCR库未安装，无法进行文本提取")
                return f"图像过大无法分析，且OCR功能未配置。建议：\n1. 缩小图像文件\n2. 转换为文档格式\n3. 手动提取关键信息\n\n文件：{file_path.name}"
            
            # 打开图像并进行OCR
            with Image.open(file_path) as img:
                # 转换为灰度以提高OCR准确性
                if img.mode != 'L':
                    img = img.convert('L')
                
                # 使用中文OCR配置
                extracted_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                
                if extracted_text.strip():
                    # 清理和格式化提取的文本
                    lines = extracted_text.strip().split('\n')
                    cleaned_lines = [line.strip() for line in lines if line.strip()]
                    cleaned_text = '\n'.join(cleaned_lines)
                    
                    # 限制文本长度
                    max_text_length = 1500
                    if len(cleaned_text) > max_text_length:
                        cleaned_text = cleaned_text[:max_text_length] + "\n\n[注：文本已截断]"
                    
                    logger.info(f"OCR文本提取成功，提取文本长度: {len(cleaned_text)} 字符")
                    return f"图像OCR文本提取结果：\n{cleaned_text}\n\n[注：这是从图像中提取的文本，可能存在识别错误]"
                else:
                    return f"图像OCR未能提取到文本内容。建议：\n1. 确保图像清晰度\n2. 增加图像对比度\n3. 手动输入关键信息\n\n文件：{file_path.name}"
                    
        except Exception as e:
            logger.error(f"OCR文本提取失败: {e}")
            return f"图像内容过大且OCR提取失败: {str(e)}。建议：\n1. 使用更小的图像文件\n2. 转换为文档格式\n3. 手动提取关键信息\n\n文件：{file_path.name}"
    
    async def _process_medical_pdf(self, file_path: Path) -> str:
        """处理医疗PDF文档，支持文本提取和OCR策略"""
        try:
            doc = fitz.open(file_path)
            full_text = ""
            ocr_pages = []  # 记录需要OCR的页面
            
            # 第一步：尝试直接文本提取
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                
                if text:
                    # 有文本内容，直接使用
                    full_text += f"\n--- 第{page_num + 1}页 ---\n{text}"
                    logger.info(f"第{page_num + 1}页：直接文本提取成功 ({len(text)} 字符)")
                else:
                    # 无文本内容，标记为需要OCR
                    ocr_pages.append(page_num)
                    logger.info(f"第{page_num + 1}页：无文本内容，需要OCR处理")
            
            # 第二步：对需要OCR的页面进行处理
            if ocr_pages:
                logger.info(f"开始OCR处理 {len(ocr_pages)} 个页面...")
                ocr_text = await self._ocr_pdf_pages(doc, ocr_pages)
                if ocr_text:
                    full_text += f"\n--- OCR提取内容 ---\n{ocr_text}"
            
            doc.close()
            
            # 第三步：如果完全没有内容，尝试图像分析
            if not full_text.strip():
                logger.info("PDF无文本内容，尝试图像分析...")
                return await self._analyze_pdf_as_images(file_path)
            
            # 第四步：分析提取的文本
            return await self._analyze_pdf_text(full_text, len(ocr_pages) > 0)
            
        except Exception as e:
            logger.error(f"处理PDF文档时出错: {e}")
            return f"PDF处理失败: {str(e)}"
    
    async def _ocr_pdf_pages(self, doc, page_numbers: List[int]) -> str:
        """对PDF页面进行OCR文本提取"""
        try:
            # 检查OCR依赖
            try:
                import pytesseract  # type: ignore[import]
            except ImportError:
                logger.warning("OCR库未安装，跳过OCR处理")
                return "OCR功能未配置，无法提取扫描页面文本"
            
            ocr_text = ""
            max_pages_to_ocr = 5  # 限制OCR页面数量以控制处理时间
            
            for i, page_num in enumerate(page_numbers[:max_pages_to_ocr]):
                try:
                    page = doc.load_page(page_num)
                    
                    # 将PDF页面转换为图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2倍缩放提高OCR质量
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # 使用PIL处理图像
                    img = Image.open(io.BytesIO(img_data))
                    
                    # 图像预处理以提高OCR准确性
                    img = self._preprocess_image_for_ocr(img)
                    
                    # 执行OCR
                    page_text = pytesseract.image_to_string(
                        img, 
                        lang='chi_sim+eng',  # 中英文识别
                        config='--psm 6'     # 假设单一文本块
                    ).strip()
                    
                    if page_text:
                        ocr_text += f"\n--- 第{page_num + 1}页 (OCR) ---\n{page_text}"
                        logger.info(f"第{page_num + 1}页OCR成功 ({len(page_text)} 字符)")
                    else:
                        logger.info(f"第{page_num + 1}页OCR未提取到文本")
                        
                except Exception as page_e:
                    logger.warning(f"第{page_num + 1}页OCR失败: {page_e}")
                    continue
            
            if len(page_numbers) > max_pages_to_ocr:
                ocr_text += f"\n[注：仅处理前{max_pages_to_ocr}页OCR，剩余{len(page_numbers) - max_pages_to_ocr}页已跳过]"
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"OCR处理失败: {e}")
            return f"OCR处理失败: {str(e)}"
    
    def _preprocess_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """图像预处理以提高OCR准确性"""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # 转换为灰度
            if img.mode != 'L':
                img = img.convert('L')
            
            # 增强对比度
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.5)
            
            # 锐化
            img = img.filter(ImageFilter.SHARPEN)
            
            # 去噪
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            return img
            
        except Exception as e:
            logger.warning(f"图像预处理失败: {e}")
            return img
    
    async def _analyze_pdf_as_images(self, file_path: Path) -> str:
        """将PDF作为图像进行AI分析（最后的备用方案）"""
        try:
            doc = fitz.open(file_path)
            
            # 只分析前几页以控制成本和时间
            max_pages = 3
            analysis_results = []
            
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                
                # 转换为图像
                mat = fitz.Matrix(1.5, 1.5)  # 适中的分辨率
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # 压缩图像
                img = Image.open(io.BytesIO(img_data))
                compressed_data = await self._compress_image_data(img, target_size=20000)
                base64_image = base64.b64encode(compressed_data).decode('utf-8')
                data_url = f"data:image/png;base64,{base64_image}"
                
                # AI分析
                messages = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"分析PDF第{page_num + 1}页的医疗内容，提取关键信息："},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                ]
                
                response = await self.vision_model.ainvoke(messages)
                analysis_results.append(f"第{page_num + 1}页分析：\n{response.content}")
            
            doc.close()
            
            combined_analysis = "\n\n".join(analysis_results)
            if len(doc) > max_pages:
                combined_analysis += f"\n\n[注：仅分析前{max_pages}页，总共{len(doc)}页]"
            
            return f"PDF图像分析结果：\n{combined_analysis}"
            
        except Exception as e:
            logger.error(f"PDF图像分析失败: {e}")
            return f"PDF图像分析失败: {str(e)}"
    
    async def _compress_image_data(self, img: Image.Image, target_size: int) -> bytes:
        """压缩PIL图像到指定大小"""
        import io
        
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 自适应压缩
        quality = 85
        for _ in range(5):
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            data = output.getvalue()
            
            if len(data) <= target_size:
                return data
            
            quality -= 15
            if quality < 20:
                break
        
        return data
    
    async def _analyze_pdf_text(self, full_text: str, has_ocr_content: bool) -> str:
        """分析PDF提取的文本内容"""
        try:
            # 限制文本长度以避免超过API字符限制
            max_text_length = 3000
            text_for_analysis = full_text[:max_text_length]
            
            # 构建分析提示词
            ocr_note = " (包含OCR提取内容)" if has_ocr_content else ""
            analysis_prompt = f"""分析医疗PDF文档{ocr_note}，提取关键信息：

文档内容：
{text_for_analysis}

请简要提供：
1. 文档类型（检查报告/病历/处方等）
2. 关键医疗信息
3. 异常发现或重要指标
4. 医生建议或诊断结论"""

            # 检查总字符数
            total_chars = len(analysis_prompt)
            logger.info(f"PDF分析提示词总字符数: {total_chars}")
            
            if total_chars > 40000:
                # 进一步缩短文本
                text_for_analysis = full_text[:2500]
                analysis_prompt = f"""分析医疗文档{ocr_note}：

{text_for_analysis}

简要说明：文档类型和主要发现。"""
                logger.info(f"缩短后字符数: {len(analysis_prompt)}")

            messages = [HumanMessage(content=analysis_prompt)]
            response = await self.vision_model.ainvoke(messages)
            
            # 返回分析结果和部分原始文本
            result = f"PDF分析{ocr_note}：\n{response.content}"
            
            # 添加原始文本摘要
            if len(full_text) > 1000:
                result += f"\n\n原始文本摘要：\n{full_text[:1000]}..."
            else:
                result += f"\n\n原始文本：\n{full_text}"
            
            return result
            
        except Exception as e:
            logger.error(f"PDF文本分析失败: {e}")
            return f"PDF文本分析失败: {str(e)}\n\n原始文本：\n{full_text[:1000]}..."
    
    async def _process_medical_document(self, file_path: Path) -> str:
        """处理其他医疗文档"""
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # 限制内容长度，避免超过API字符限制
                max_content_length = 2000
                if len(content) > max_content_length:
                    content = content[:max_content_length]
                    logger.info(f"文档内容已截断至{max_content_length}字符")
                    
                return f"文档内容：\n{content}..."
            else:
                # 对于docx等格式，可以使用python-docx库
                return f"暂不支持 {file_path.suffix} 格式的详细解析，建议转换为PDF或TXT格式"
            
        except Exception as e:
            logger.error(f"处理文档时出错: {e}")
            return f"文档处理失败: {str(e)}"
    
    def get_attachment_summary(self, attachment_info: Dict[str, Any]) -> str:
        """获取附件的简要摘要用于对话"""
        if attachment_info["status"] == "error":
            return f"附件 {attachment_info['original_filename']} 处理失败"
        
        # 检查数据来源并添加指示器
        cache_indicator = ""
        status = attachment_info.get("status", "")
        if status == "success_from_cache":
            cache_indicator = " [来自缓存]"
        elif status == "success_from_user_data":
            cache_indicator = " [来自用户数据]"
        elif status == "success_from_previous_data":
            cache_indicator = " [继续使用之前的分析]"
        
        content = attachment_info["extracted_content"]
        # 提取前200个字符作为摘要
        summary = content[:200] + "..." if len(content) > 200 else content
        
        return f"医疗附件 '{attachment_info['original_filename']}' 分析结果{cache_indicator}：{summary}"
    
    def clear_user_medical_cache(self, user_ip: str) -> int:
        """
        清理特定用户的医疗附件缓存
        
        Args:
            user_ip: 用户IP
            
        Returns:
            清理的缓存条目数量
        """
        return self.cache_manager.clear_user_cache(user_ip)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取医疗附件缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        return self.cache_manager.get_cache_stats()
    
    def is_attachment_cached(self, user_ip: str, filename: str) -> bool:
        """
        检查医疗附件是否已缓存
        
        Args:
            user_ip: 用户IP
            filename: 文件名
            
        Returns:
            是否已缓存
        """
        cached_result = self.cache_manager.get_cached_analysis(user_ip, filename)
        return cached_result is not None

# 全局处理器实例
medical_processor = MedicalAttachmentProcessor()