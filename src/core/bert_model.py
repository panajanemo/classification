import torch
import logging
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from config.settings import settings


logger = logging.getLogger(__name__)


class BERTModel:
    """BERT模型封装类"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化BERT模型
        
        Args:
            model_name: 模型名称
            device: 设备类型 (cpu/cuda/mps/auto)
        """
        self.model_name = model_name or settings.model_name
        self.device = self._get_optimal_device(device or settings.device)
        self.max_length = settings.max_length
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
        logger.info(f"初始化BERT模型: {self.model_name}, 设备: {self.device}")
    
    def _get_optimal_device(self, device: str) -> str:
        """
        获取最优设备
        
        Args:
            device: 指定的设备或"auto"
            
        Returns:
            最优设备名称
        """
        if device != "auto":
            return device
        
        # 自动选择最佳设备
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> None:
        """加载模型和分词器"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir
            )
            
            # 移动到指定设备
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载BERT模型: {e}")
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的张量
        """
        if not self._is_loaded:
            self.load_model()
        
        if not texts:
            return torch.empty(0, self.model.config.hidden_size)
        
        try:
            # 分词和编码
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到指定设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**encoded)
                # 使用[CLS]标记的隐藏状态作为句子表示
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise RuntimeError(f"文本编码错误: {e}")
    
    def encode_text_pair(self, text1: str, text2: str) -> torch.Tensor:
        """
        编码文本对
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            编码后的张量
        """
        if not self._is_loaded:
            self.load_model()
        
        try:
            # 使用分词器的文本对编码功能
            encoded = self.tokenizer(
                text1,
                text2,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到指定设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**encoded)
                # 使用[CLS]标记的隐藏状态
                embedding = outputs.last_hidden_state[0, 0, :]
            
            return embedding
            
        except Exception as e:
            logger.error(f"文本对编码失败: {e}")
            raise RuntimeError(f"文本对编码错误: {e}")
    
    def calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
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
            cos_sim = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            )
            return cos_sim.item()
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def batch_similarities(self, embeddings: torch.Tensor) -> List[float]:
        """
        批量计算相邻嵌入向量的相似度
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            相似度列表
        """
        if embeddings.size(0) < 2:
            return []
        
        similarities = []
        for i in range(embeddings.size(0) - 1):
            sim = self.calculate_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        return similarities
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_device(self) -> str:
        """获取当前设备"""
        return self.device
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "is_loaded": self._is_loaded
        }