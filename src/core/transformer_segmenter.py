import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector, TextType
from src.utils.text_processor import TextProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class SegmentationTransformer(nn.Module):
    """
    上层Transformer分段模型 - 基于Transformer²架构
    在Sentence Transformer基础上添加段落级别的学习
    """
    
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, 
                 num_layers: int = 2, dropout: float = 0.1):
        """
        初始化Transformer分段模型
        
        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 段落边界预测头
        self.boundary_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 语义一致性预测头
        self.consistency_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Transformer分段模型初始化完成: {embedding_dim}维, {num_heads}头, {num_layers}层")
    
    def forward(self, sentence_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            sentence_embeddings: 句子嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            包含边界预测和一致性分数的字典
        """
        batch_size, seq_len, _ = sentence_embeddings.shape
        
        # 添加位置编码
        x = self.positional_encoding(sentence_embeddings)
        
        # Transformer编码
        if attention_mask is not None:
            # 转换掩码格式 (True表示有效位置)
            transformer_mask = attention_mask.bool()
        else:
            transformer_mask = None
            
        encoded = self.transformer(x, src_key_padding_mask=~transformer_mask if transformer_mask is not None else None)
        
        # 边界预测 (对每个相邻句子对预测是否应该分段)
        boundary_scores = []
        for i in range(seq_len - 1):
            boundary_score = self.boundary_predictor(encoded[:, i, :])
            boundary_scores.append(boundary_score)
        
        if boundary_scores:
            boundary_predictions = torch.cat(boundary_scores, dim=1)  # [batch_size, seq_len-1]
        else:
            boundary_predictions = torch.zeros(batch_size, 0, device=sentence_embeddings.device)
        
        # 语义一致性预测
        consistency_scores = []
        for i in range(seq_len - 1):
            # 相邻句子对的连接表示
            pair_repr = torch.cat([encoded[:, i, :], encoded[:, i+1, :]], dim=-1)
            consistency_score = self.consistency_predictor(pair_repr)
            consistency_scores.append(consistency_score)
        
        if consistency_scores:
            consistency_predictions = torch.cat(consistency_scores, dim=1)  # [batch_size, seq_len-1]
        else:
            consistency_predictions = torch.zeros(batch_size, 0, device=sentence_embeddings.device)
        
        return {
            'encoded_sentences': encoded,
            'boundary_predictions': boundary_predictions,
            'consistency_predictions': consistency_predictions
        }


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerSemanticSegmenter:
    """
    基于Transformer²的语义分段器
    结合Sentence Transformer和上层Transformer实现更精准的分段
    """
    
    def __init__(self, 
                 sentence_model: Optional[SentenceTransformerModel] = None,
                 text_processor: Optional[TextProcessor] = None,
                 type_detector: Optional[TextTypeDetector] = None,
                 device: str = "auto"):
        """
        初始化Transformer语义分段器
        
        Args:
            sentence_model: Sentence Transformer模型实例
            text_processor: 文本处理器实例
            type_detector: 文本类型检测器实例
            device: 计算设备
        """
        self.sentence_model = sentence_model or SentenceTransformerModel(device=device)
        self.text_processor = text_processor or TextProcessor()
        self.type_detector = type_detector or TextTypeDetector()
        
        # 自动设备选择
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # 确保sentence model已加载
        if not self.sentence_model.is_loaded():
            self.sentence_model.load_model()
        
        # 初始化Transformer分段模型
        embedding_dim = self.sentence_model.model.get_sentence_embedding_dimension()
        self.transformer_model = SegmentationTransformer(
            embedding_dim=embedding_dim,
            num_heads=6,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
        
        # 设置为评估模式（当前不进行训练）
        self.transformer_model.eval()
        
        # 默认配置
        self.default_config = {
            "boundary_threshold": 0.3,      # 边界预测阈值（降低以便更容易分段）
            "consistency_threshold": 0.7,   # 一致性阈值（提高以便更严格）
            "min_paragraph_length": settings.min_paragraph_length,
            "max_paragraph_length": settings.max_paragraph_length,
            "combine_predictions": True,    # 是否结合传统方法
            "alpha": 0.3,                  # Transformer预测权重（降低，因为未训练）
            "beta": 0.7                    # 传统方法权重（提高，更可靠）
        }
        
        logger.info(f"Transformer语义分段器初始化完成，设备: {self.device}")
    
    def _prepare_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """准备句子嵌入"""
        embeddings = self.sentence_model.encode_texts(sentences)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        return embeddings.to(self.device).unsqueeze(0)  # 添加batch维度
    
    def _traditional_similarity_analysis(self, embeddings: np.ndarray) -> List[float]:
        """传统相似度分析作为基线"""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.sentence_model.calculate_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        return similarities
    
    def _combine_predictions(self, transformer_boundaries: List[float], 
                           traditional_similarities: List[float],
                           alpha: float = 0.7) -> List[float]:
        """结合Transformer预测和传统方法"""
        if len(transformer_boundaries) != len(traditional_similarities):
            # 长度不匹配时使用Transformer结果
            return transformer_boundaries
        
        combined = []
        for t_bound, t_sim in zip(transformer_boundaries, traditional_similarities):
            # 边界分数 = alpha * Transformer边界预测 + (1-alpha) * (1 - 传统相似度)
            boundary_score = alpha * t_bound + (1 - alpha) * (1 - t_sim)
            combined.append(boundary_score)
        
        return combined
    
    def _find_segment_boundaries(self, boundary_scores: List[float], 
                               consistency_scores: List[float],
                               config: Dict[str, Any]) -> List[int]:
        """根据预测分数确定分段边界"""
        if not boundary_scores:
            return []
        
        boundaries = []
        boundary_threshold = config["boundary_threshold"]
        consistency_threshold = config["consistency_threshold"]
        
        for i, (b_score, c_score) in enumerate(zip(boundary_scores, consistency_scores)):
            # 边界条件：边界分数高且一致性分数低
            if b_score > boundary_threshold and c_score < consistency_threshold:
                boundaries.append(i + 1)  # +1因为这是句子i和i+1之间的边界
        
        return boundaries
    
    def _optimize_boundaries_by_length(self, sentences: List[str], 
                                     boundaries: List[int],
                                     config: Dict[str, Any]) -> List[int]:
        """根据段落长度优化边界"""
        if not boundaries:
            return boundaries
        
        optimized = []
        current_start = 0
        
        for boundary in boundaries:
            current_paragraph = " ".join(sentences[current_start:boundary])
            paragraph_length = len(current_paragraph)
            
            if paragraph_length >= config["min_paragraph_length"]:
                if paragraph_length <= config["max_paragraph_length"]:
                    optimized.append(boundary)
                else:
                    # 段落太长，尝试在中间分割
                    sentences_in_para = boundary - current_start
                    if sentences_in_para > 2:
                        mid_point = current_start + sentences_in_para // 2
                        optimized.append(mid_point)
                    optimized.append(boundary)
            # 段落太短则跳过此边界
            
            current_start = boundary
        
        return optimized
    
    def segment_text_transformer(self, text: str, 
                               custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        使用Transformer²架构进行文本分段
        
        Args:
            text: 输入文本
            custom_config: 自定义配置
            
        Returns:
            分段结果
        """
        try:
            # 输入验证
            if not self.text_processor.validate_input(text):
                return {"error": "输入文本无效"}
            
            # 检测文本类型
            text_type, type_info = self.type_detector.detect_text_type(text)
            
            # 合并配置
            final_config = {**self.default_config}
            if custom_config:
                final_config.update(custom_config)
            
            # 预处理和分句
            normalized_text = self.text_processor.normalize_text(text)
            sentences = self.text_processor.split_sentences(normalized_text)
            
            if len(sentences) < 2:
                return {
                    "paragraphs": [{"text": normalized_text, "type": "content"}] if normalized_text else [],
                    "text_type": text_type.value,
                    "method": "transformer",
                    "config_used": final_config,
                    "quality": {"message": "文本太短，无需分段"}
                }
            
            # 获取句子嵌入
            sentence_embeddings = self._prepare_embeddings(sentences)
            
            # Transformer预测
            with torch.no_grad():
                predictions = self.transformer_model(sentence_embeddings)
                
                boundary_predictions = predictions['boundary_predictions'].cpu().numpy().flatten()
                consistency_predictions = predictions['consistency_predictions'].cpu().numpy().flatten()
            
            # 传统方法作为对比
            traditional_similarities = self._traditional_similarity_analysis(
                self.sentence_model.encode_texts(sentences)
            )
            
            # 结合预测结果
            if final_config["combine_predictions"] and len(traditional_similarities) == len(boundary_predictions):
                combined_boundaries = self._combine_predictions(
                    boundary_predictions.tolist(), 
                    traditional_similarities,
                    final_config["alpha"]
                )
            else:
                combined_boundaries = boundary_predictions.tolist()
            
            # 确定最终边界
            boundaries = self._find_segment_boundaries(
                combined_boundaries, 
                consistency_predictions.tolist(),
                final_config
            )
            
            # 长度优化
            optimized_boundaries = self._optimize_boundaries_by_length(
                sentences, boundaries, final_config
            )
            
            # 构建段落
            paragraphs = []
            all_boundaries = [0] + sorted(optimized_boundaries) + [len(sentences)]
            
            for i in range(len(all_boundaries) - 1):
                start_idx = all_boundaries[i]
                end_idx = all_boundaries[i + 1]
                
                if start_idx < end_idx:
                    paragraph_sentences = sentences[start_idx:end_idx]
                    paragraph_text = " ".join(paragraph_sentences)
                    paragraph_text = self.text_processor.normalize_text(paragraph_text)
                    
                    if paragraph_text.strip():
                        paragraphs.append({
                            "text": paragraph_text.strip(),
                            "type": "content",
                            "length": len(paragraph_text),
                            "sentence_count": len(paragraph_sentences),
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "method": "transformer"
                        })
            
            # 评估质量
            quality_metrics = self._evaluate_transformer_quality(
                paragraphs, boundary_predictions, consistency_predictions
            )
            
            return {
                "paragraphs": paragraphs,
                "text_type": text_type.value,
                "type_confidence": type_info["confidence"],
                "method": "transformer",
                "config_used": final_config,
                "sentence_count": len(sentences),
                "boundary_count": len(optimized_boundaries),
                "transformer_predictions": {
                    "boundary_scores": boundary_predictions.tolist(),
                    "consistency_scores": consistency_predictions.tolist(),
                    "combined_scores": combined_boundaries
                },
                "quality": quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Transformer分段失败: {e}")
            return {"error": f"分段处理错误: {str(e)}"}
    
    def _evaluate_transformer_quality(self, paragraphs: List[Dict[str, Any]], 
                                    boundary_predictions: np.ndarray,
                                    consistency_predictions: np.ndarray) -> Dict[str, Any]:
        """评估Transformer分段质量"""
        if not paragraphs:
            return {"error": "无段落可评估"}
        
        # 基本统计
        lengths = [p["length"] for p in paragraphs]
        sentence_counts = [p["sentence_count"] for p in paragraphs]
        
        # Transformer特定指标
        avg_boundary_confidence = float(np.mean(boundary_predictions)) if len(boundary_predictions) > 0 else 0.0
        avg_consistency_score = float(np.mean(consistency_predictions)) if len(consistency_predictions) > 0 else 0.0
        
        # 预测分布
        boundary_std = float(np.std(boundary_predictions)) if len(boundary_predictions) > 0 else 0.0
        consistency_std = float(np.std(consistency_predictions)) if len(consistency_predictions) > 0 else 0.0
        
        # 综合质量分数
        quality_score = self._calculate_transformer_quality_score(
            len(paragraphs), np.mean(lengths), avg_boundary_confidence, 
            avg_consistency_score, boundary_std
        )
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_length": round(np.mean(lengths), 2),
            "length_std": round(np.std(lengths), 2),
            "avg_sentence_count": round(np.mean(sentence_counts), 2),
            "avg_boundary_confidence": round(avg_boundary_confidence, 3),
            "avg_consistency_score": round(avg_consistency_score, 3),
            "boundary_prediction_std": round(boundary_std, 3),
            "consistency_prediction_std": round(consistency_std, 3),
            "quality_score": round(quality_score, 3),
            "method": "transformer"
        }
    
    def _calculate_transformer_quality_score(self, count: int, avg_length: float,
                                           boundary_conf: float, consistency: float,
                                           prediction_std: float) -> float:
        """计算Transformer质量分数"""
        score = 1.0
        
        # 边界预测置信度奖励
        score *= (1 + boundary_conf * 0.3)
        
        # 一致性分数（低一致性意味着好的分段）
        score *= (1 + (1 - consistency) * 0.2)
        
        # 预测稳定性（标准差不能太高）
        if prediction_std > 0.3:
            score *= 0.9
        
        # 段落数量合理性
        if count < 2:
            score *= 0.8
        elif count > 15:
            score *= 0.9
        
        # 平均长度合理性
        if 100 <= avg_length <= 500:
            score *= 1.1
        elif avg_length < 50:
            score *= 0.8
        
        return max(0.0, min(1.0, score))