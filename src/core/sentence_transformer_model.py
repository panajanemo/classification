import torch
import logging
import numpy as np
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from config.settings import settings


logger = logging.getLogger(__name__)


class SentenceTransformerModel:
    """Sentence Transformer模型封装类 - 针对RAG场景优化"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化Sentence Transformer模型
        
        Args:
            model_name: 模型名称
            device: 设备类型 (cpu/cuda/mps/auto)
        """
        # 默认使用轻量级但性能优秀的模型
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.device = self._get_optimal_device(device or settings.device)
        
        # 模型实例
        self.model = None
        self._is_loaded = False
        
        # 性能配置
        self.batch_size = getattr(settings, 'batch_size', 32)
        self.max_seq_length = getattr(settings, 'max_length', 512)
        
        # 缓存配置
        self.enable_cache = True
        self.cache_size = 1000  # 最大缓存条目数
        self.embeddings_cache = {}  # 嵌入向量缓存
        
        logger.info(f"初始化Sentence Transformer模型: {self.model_name}, 设备: {self.device}")
    
    def _get_optimal_device(self, device: str) -> str:
        """获取最优设备"""
        if device != "auto":
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """生成文本列表的缓存键"""
        # 使用文本内容的哈希值作为缓存键
        text_content = "|".join(sorted(texts))  # 排序确保一致性
        return hashlib.md5(text_content.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从缓存获取嵌入向量"""
        if not self.enable_cache:
            return None
        return self.embeddings_cache.get(cache_key)
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """保存嵌入向量到缓存"""
        if not self.enable_cache:
            return
        
        # 如果缓存已满，删除最旧的条目
        if len(self.embeddings_cache) >= self.cache_size:
            # 简单的LRU策略：删除第一个条目
            oldest_key = next(iter(self.embeddings_cache))
            del self.embeddings_cache[oldest_key]
        
        self.embeddings_cache[cache_key] = embeddings.copy()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.embeddings_cache.clear()
        logger.info("嵌入向量缓存已清空")
    
    def load_model(self) -> None:
        """加载模型"""
        try:
            logger.info(f"正在加载Sentence Transformer模型: {self.model_name}")
            
            # 加载sentence-transformers模型
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=settings.model_cache_dir
            )
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            self._is_loaded = True
            logger.info(f"模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载Sentence Transformer模型: {e}")
    
    def encode_texts(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            normalize_embeddings: 是否归一化嵌入向量
            
        Returns:
            编码后的嵌入矩阵
        """
        if not self._is_loaded:
            self.load_model()
        
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(texts)
            cached_embeddings = self._get_from_cache(cache_key)
            
            if cached_embeddings is not None:
                logger.debug(f"从缓存获取嵌入向量，文本数量: {len(texts)}")
                return cached_embeddings
            
            # 使用sentence-transformers进行编码
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # 保存到缓存
            self._save_to_cache(cache_key, embeddings)
            logger.debug(f"计算并缓存嵌入向量，文本数量: {len(texts)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise RuntimeError(f"文本编码错误: {e}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个嵌入向量的相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度值
        """
        try:
            # 计算余弦相似度
            if len(embedding1.shape) == 1:
                embedding1 = embedding1.reshape(1, -1)
            if len(embedding2.shape) == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            # 使用numpy计算余弦相似度
            similarity = np.dot(embedding1, embedding2.T) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity.item() if hasattr(similarity, 'item') else similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def batch_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        批量计算相邻嵌入向量的相似度
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            相似度列表
        """
        if embeddings.shape[0] < 2:
            return []
        
        similarities = []
        for i in range(embeddings.shape[0] - 1):
            sim = self.calculate_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        return similarities
    
    def calculate_semantic_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算全局语义相似度矩阵
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            相似度矩阵
        """
        try:
            # 计算所有句子对之间的相似度
            similarity_matrix = np.dot(embeddings, embeddings.T)
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"相似度矩阵计算失败: {e}")
            return np.zeros((embeddings.shape[0], embeddings.shape[0]))
    
    def find_semantic_boundaries(self, texts: List[str], window_size: int = 3, 
                                threshold: float = 0.5) -> List[int]:
        """
        使用滑动窗口方法找到语义边界
        
        Args:
            texts: 文本列表
            window_size: 窗口大小
            threshold: 相似度阈值
            
        Returns:
            边界位置列表
        """
        if len(texts) < window_size * 2:
            return []
        
        try:
            embeddings = self.encode_texts(texts)
            boundaries = []
            
            for i in range(window_size, len(texts) - window_size):
                # 计算前窗口和后窗口的平均相似度
                left_window = embeddings[i-window_size:i]
                right_window = embeddings[i:i+window_size]
                
                # 计算窗口内的平均嵌入
                left_centroid = np.mean(left_window, axis=0)
                right_centroid = np.mean(right_window, axis=0)
                
                # 计算窗口间相似度
                similarity = self.calculate_similarity(left_centroid, right_centroid)
                
                # 如果相似度低于阈值，认为是边界
                if similarity < threshold:
                    boundaries.append(i)
            
            return boundaries
            
        except Exception as e:
            logger.error(f"语义边界检测失败: {e}")
            return []
    
    def extract_key_phrases(self, texts: List[str], top_k: int = 5) -> List[List[str]]:
        """
        为每个文本段落提取关键短语（基于语义相似度）
        
        Args:
            texts: 文本列表
            top_k: 每个段落返回的关键短语数量
            
        Returns:
            每个文本对应的关键短语列表
        """
        try:
            # 简单的关键词提取实现
            # 这里可以后续改进为更复杂的算法
            key_phrases = []
            
            for text in texts:
                # 基本的关键词提取（可以后续优化）
                words = text.split()
                # 过滤停用词和短词
                filtered_words = [w for w in words if len(w) > 2 and w.isalpha()]
                # 取前top_k个词作为关键短语
                phrases = filtered_words[:top_k] if len(filtered_words) >= top_k else filtered_words
                key_phrases.append(phrases)
            
            return key_phrases
            
        except Exception as e:
            logger.error(f"关键短语提取失败: {e}")
            return [[] for _ in texts]
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_device(self) -> str:
        """获取当前设备"""
        return self.device
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "model_type": "sentence-transformers"
        }
        
        if self._is_loaded and self.model:
            info.update({
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
            })
        
        return info