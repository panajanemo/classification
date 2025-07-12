import logging
from typing import List, Tuple, Optional
import torch
import numpy as np
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class SemanticSegmenter:
    """语义分段器"""
    
    def __init__(self, bert_model: Optional[BERTModel] = None, 
                 text_processor: Optional[TextProcessor] = None):
        """
        初始化语义分段器
        
        Args:
            bert_model: BERT模型实例
            text_processor: 文本处理器实例
        """
        self.bert_model = bert_model or BERTModel()
        self.text_processor = text_processor or TextProcessor()
        self.threshold = settings.threshold
        self.min_paragraph_length = settings.min_paragraph_length
        self.max_paragraph_length = settings.max_paragraph_length
        
        logger.info(f"语义分段器初始化完成，阈值: {self.threshold}")
    
    def calculate_continuation_probabilities(self, sentences: List[str]) -> List[float]:
        """
        计算句子间的续写概率
        
        Args:
            sentences: 句子列表
            
        Returns:
            续写概率列表 (长度为len(sentences)-1)
        """
        if len(sentences) < 2:
            return []
        
        try:
            # 编码所有句子
            embeddings = self.bert_model.encode_texts(sentences)
            
            # 计算相邻句子的相似度作为续写概率
            probabilities = []
            for i in range(len(sentences) - 1):
                similarity = self.bert_model.calculate_similarity(
                    embeddings[i], embeddings[i + 1]
                )
                probabilities.append(similarity)
            
            logger.debug(f"计算了 {len(probabilities)} 个续写概率")
            return probabilities
            
        except Exception as e:
            logger.error(f"续写概率计算失败: {e}")
            return [0.0] * (len(sentences) - 1)
    
    def identify_paragraph_boundaries(self, sentences: List[str], 
                                    probabilities: List[float]) -> List[int]:
        """
        识别段落边界
        
        Args:
            sentences: 句子列表
            probabilities: 续写概率列表
            
        Returns:
            段落边界索引列表
        """
        if not probabilities:
            return []
        
        boundaries = []
        
        # 基于阈值的简单分段
        for i, prob in enumerate(probabilities):
            if prob < self.threshold:
                boundaries.append(i + 1)  # 在第i+1个句子前分段
        
        # 考虑段落长度约束的优化
        boundaries = self._optimize_boundaries_by_length(sentences, boundaries)
        
        logger.debug(f"识别出 {len(boundaries)} 个段落边界")
        return boundaries
    
    def _optimize_boundaries_by_length(self, sentences: List[str], 
                                     boundaries: List[int]) -> List[int]:
        """
        根据段落长度约束优化边界
        
        Args:
            sentences: 句子列表
            boundaries: 原始边界列表
            
        Returns:
            优化后的边界列表
        """
        if not boundaries:
            return boundaries
        
        optimized = []
        current_length = 0
        current_start = 0
        
        for boundary in boundaries:
            # 计算当前段落长度
            paragraph_text = " ".join(sentences[current_start:boundary])
            paragraph_length = len(paragraph_text)
            
            # 如果段落太短，尝试合并
            if paragraph_length < self.min_paragraph_length and optimized:
                continue  # 跳过这个边界，与下个段落合并
            
            # 如果段落太长，尝试在内部分割
            if paragraph_length > self.max_paragraph_length:
                # 在中间位置强制分割
                mid_point = current_start + (boundary - current_start) // 2
                optimized.append(mid_point)
            
            optimized.append(boundary)
            current_start = boundary
        
        return optimized
    
    def build_paragraphs(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        构建段落
        
        Args:
            sentences: 句子列表
            boundaries: 段落边界列表
            
        Returns:
            段落列表
        """
        if not sentences:
            return []
        
        paragraphs = []
        start = 0
        
        # 添加段落边界0和最后位置
        all_boundaries = [0] + sorted(boundaries) + [len(sentences)]
        
        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]
            
            if start_idx < end_idx:
                paragraph_sentences = sentences[start_idx:end_idx]
                paragraph = " ".join(paragraph_sentences)
                
                # 清理段落文本
                paragraph = self.text_processor.normalize_text(paragraph)
                
                if paragraph.strip():
                    paragraphs.append(paragraph.strip())
        
        logger.debug(f"构建了 {len(paragraphs)} 个段落")
        return paragraphs
    
    def evaluate_segmentation_quality(self, paragraphs: List[str]) -> dict:
        """
        评估分段质量
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            质量评估指标
        """
        if not paragraphs:
            return {"error": "无段落可评估"}
        
        # 计算基本统计信息
        lengths = [len(p) for p in paragraphs]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # 计算长度分布合理性
        too_short = sum(1 for l in lengths if l < self.min_paragraph_length)
        too_long = sum(1 for l in lengths if l > self.max_paragraph_length)
        
        # 计算段落间语义一致性（如果有多个段落）
        semantic_consistency = 0.0
        if len(paragraphs) > 1:
            try:
                embeddings = self.bert_model.encode_texts(paragraphs)
                similarities = self.bert_model.batch_similarities(embeddings)
                semantic_consistency = np.mean(similarities) if similarities else 0.0
            except Exception:
                semantic_consistency = 0.0
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_length": round(avg_length, 2),
            "length_std": round(std_length, 2),
            "too_short_count": too_short,
            "too_long_count": too_long,
            "semantic_consistency": round(semantic_consistency, 3),
            "quality_score": self._calculate_quality_score(
                len(paragraphs), avg_length, too_short, too_long, semantic_consistency
            )
        }
    
    def _calculate_quality_score(self, count: int, avg_length: float, 
                               too_short: int, too_long: int, 
                               consistency: float) -> float:
        """计算综合质量分数"""
        score = 1.0
        
        # 长度合理性惩罚
        if too_short > 0:
            score -= 0.1 * too_short / count
        if too_long > 0:
            score -= 0.1 * too_long / count
        
        # 平均长度合理性
        ideal_length = (self.min_paragraph_length + self.max_paragraph_length) / 2
        length_ratio = min(avg_length, ideal_length) / max(avg_length, ideal_length)
        score *= length_ratio
        
        # 语义一致性奖励
        score *= (1 + consistency * 0.2)
        
        return max(0.0, min(1.0, score))
    
    def segment_text(self, text: str) -> dict:
        """
        对文本进行语义分段
        
        Args:
            text: 输入文本
            
        Returns:
            分段结果字典
        """
        try:
            # 验证输入
            if not self.text_processor.validate_input(text):
                return {"error": "输入文本无效"}
            
            # 预处理文本
            normalized_text = self.text_processor.normalize_text(text)
            
            # 分句
            sentences = self.text_processor.split_sentences(normalized_text)
            if len(sentences) < 2:
                return {
                    "paragraphs": [normalized_text] if normalized_text else [],
                    "sentence_count": len(sentences),
                    "quality": {"message": "文本太短，无需分段"}
                }
            
            # 计算续写概率
            probabilities = self.calculate_continuation_probabilities(sentences)
            
            # 识别段落边界
            boundaries = self.identify_paragraph_boundaries(sentences, probabilities)
            
            # 构建段落
            paragraphs = self.build_paragraphs(sentences, boundaries)
            
            # 评估质量
            quality = self.evaluate_segmentation_quality(paragraphs)
            
            return {
                "paragraphs": paragraphs,
                "sentence_count": len(sentences),
                "boundary_count": len(boundaries),
                "continuation_probabilities": probabilities,
                "quality": quality
            }
            
        except Exception as e:
            logger.error(f"文本分段失败: {e}")
            return {"error": f"分段处理错误: {str(e)}"}
    
    def set_threshold(self, threshold: float) -> None:
        """设置分段阈值"""
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"分段阈值已设置为: {threshold}")
        else:
            raise ValueError("阈值必须在0.0到1.0之间")