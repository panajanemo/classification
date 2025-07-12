import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector, TextType
from src.utils.text_processor import TextProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class EnhancedSemanticSegmenter:
    """增强版语义分段器 - 针对RAG场景优化"""
    
    def __init__(self, 
                 model: Optional[SentenceTransformerModel] = None,
                 text_processor: Optional[TextProcessor] = None,
                 type_detector: Optional[TextTypeDetector] = None):
        """
        初始化增强版语义分段器
        
        Args:
            model: Sentence Transformer模型实例
            text_processor: 文本处理器实例
            type_detector: 文本类型检测器实例
        """
        self.model = model or SentenceTransformerModel()
        self.text_processor = text_processor or TextProcessor()
        self.type_detector = type_detector or TextTypeDetector()
        
        # 默认配置
        self.default_config = {
            "threshold": settings.threshold,
            "window_size": 3,
            "min_paragraph_length": settings.min_paragraph_length,
            "max_paragraph_length": settings.max_paragraph_length,
            "use_structure_hints": False
        }
        
        logger.info("增强版语义分段器初始化完成")
    
    def _smooth_similarities(self, similarities: List[float], sigma: float = 1.0) -> np.ndarray:
        """
        平滑相似度序列以减少噪声
        
        Args:
            similarities: 相似度列表
            sigma: 高斯滤波器的标准差
            
        Returns:
            平滑后的相似度数组
        """
        if len(similarities) < 3:
            return np.array(similarities)
        
        similarities_array = np.array(similarities)
        smoothed = gaussian_filter1d(similarities_array, sigma=sigma)
        return smoothed
    
    def _find_semantic_valleys(self, similarities: np.ndarray, 
                             threshold: float, min_distance: int = 2) -> List[int]:
        """
        找到语义相似度的"谷底"作为分段点
        
        Args:
            similarities: 相似度数组
            threshold: 阈值
            min_distance: 最小峰值距离
            
        Returns:
            分段点位置列表
        """
        # 反转相似度以找到谷底（相似度低的地方）
        inverted_similarities = 1 - similarities
        
        # 找到低于阈值的所有点
        below_threshold = similarities < threshold
        threshold_indices = np.where(below_threshold)[0]
        
        if len(threshold_indices) == 0:
            return []
        
        # 使用峰值检测算法找到局部最小值
        try:
            peaks, _ = find_peaks(inverted_similarities, 
                                distance=min_distance,
                                height=1-threshold)
            
            # 只返回确实低于阈值的峰值
            valid_peaks = [p for p in peaks if similarities[p] < threshold]
            return valid_peaks
            
        except Exception as e:
            logger.warning(f"峰值检测失败，使用简单阈值方法: {e}")
            return threshold_indices.tolist()
    
    def _calculate_multi_scale_similarities(self, embeddings: np.ndarray, 
                                          scales: List[int] = [1, 3, 5]) -> Dict[str, np.ndarray]:
        """
        计算多尺度语义相似度
        
        Args:
            embeddings: 嵌入向量矩阵
            scales: 不同的尺度窗口大小
            
        Returns:
            不同尺度的相似度字典，所有尺度都有相同长度(len(embeddings)-1)
        """
        multi_scale_similarities = {}
        n_sentences = len(embeddings)
        
        for scale in scales:
            similarities = []
            
            # 计算每个相邻句子对之间的相似度
            for i in range(n_sentences - 1):
                if scale == 1:
                    # 直接计算相邻句子相似度
                    sim = self.model.calculate_similarity(embeddings[i], embeddings[i + 1])
                else:
                    # 多句子窗口平均相似度
                    half_window = scale // 2
                    
                    # 定义左右窗口范围
                    left_start = max(0, i - half_window)
                    left_end = i + 1
                    right_start = i
                    right_end = min(n_sentences, i + 1 + half_window)
                    
                    # 获取左右窗口的嵌入
                    left_embeddings = embeddings[left_start:left_end]
                    right_embeddings = embeddings[right_start:right_end]
                    
                    # 计算窗口质心
                    left_centroid = np.mean(left_embeddings, axis=0)
                    right_centroid = np.mean(right_embeddings, axis=0)
                    
                    # 计算质心相似度
                    sim = self.model.calculate_similarity(left_centroid, right_centroid)
                
                similarities.append(sim)
            
            multi_scale_similarities[f"scale_{scale}"] = np.array(similarities)
            logger.debug(f"Scale {scale}: 计算了 {len(similarities)} 个相似度值")
        
        return multi_scale_similarities
    
    def _combine_multi_scale_boundaries(self, multi_scale_similarities: Dict[str, np.ndarray],
                                      threshold: float, text_type: TextType) -> List[int]:
        """
        综合多尺度相似度确定最终分段边界
        
        Args:
            multi_scale_similarities: 多尺度相似度字典
            threshold: 基础阈值
            text_type: 文本类型
            
        Returns:
            最终分段边界列表
        """
        # 根据文本类型调整权重和阈值
        scale_configs = {
            TextType.TECHNICAL: {"scale_1": (0.3, 0.85), "scale_3": (0.5, 0.8), "scale_5": (0.2, 0.75)},
            TextType.NOVEL: {"scale_1": (0.2, 0.7), "scale_3": (0.3, 0.65), "scale_5": (0.5, 0.6)},
            TextType.ACADEMIC: {"scale_1": (0.4, 0.8), "scale_3": (0.4, 0.75), "scale_5": (0.2, 0.7)},
            TextType.NEWS: {"scale_1": (0.4, 0.8), "scale_3": (0.4, 0.75), "scale_5": (0.2, 0.7)},
            TextType.DIALOGUE: {"scale_1": (0.6, 0.75), "scale_3": (0.3, 0.7), "scale_5": (0.1, 0.65)},
        }
        
        configs = scale_configs.get(text_type, {"scale_1": (0.4, 0.75), "scale_3": (0.4, 0.7), "scale_5": (0.2, 0.65)})
        
        # 收集所有候选边界及其强度
        boundary_scores = {}
        
        for scale_name, similarities in multi_scale_similarities.items():
            if scale_name not in configs:
                continue
                
            weight, scale_threshold = configs[scale_name]
            
            # 平滑相似度
            smoothed_similarities = self._smooth_similarities(similarities.tolist())
            
            # 找到边界
            boundaries = self._find_semantic_valleys(smoothed_similarities, scale_threshold)
            
            logger.debug(f"{scale_name}: 阈值={scale_threshold:.3f}, 找到边界: {boundaries}")
            
            # 为每个边界计算分数
            for boundary in boundaries:
                if boundary not in boundary_scores:
                    boundary_scores[boundary] = 0.0
                
                # 边界强度 = 权重 * (1 - 相似度)
                if boundary < len(smoothed_similarities):
                    boundary_strength = weight * (1 - smoothed_similarities[boundary])
                    boundary_scores[boundary] += boundary_strength
        
        # 选择强度超过阈值的边界
        min_score = 0.05  # 降低最小边界强度阈值
        selected_boundaries = [
            boundary for boundary, score in boundary_scores.items() 
            if score >= min_score
        ]
        
        logger.info(f"边界分数: {boundary_scores}")
        logger.info(f"选择的边界: {selected_boundaries}")
        
        return sorted(selected_boundaries)
    
    def _optimize_boundaries_by_content(self, sentences: List[str], 
                                      boundaries: List[int],
                                      config: Dict[str, Any]) -> List[int]:
        """
        根据内容特征优化边界
        
        Args:
            sentences: 句子列表
            boundaries: 原始边界列表
            config: 分段配置
            
        Returns:
            优化后的边界列表
        """
        if not boundaries:
            return boundaries
        
        optimized = []
        current_start = 0
        
        for boundary in boundaries:
            # 计算当前段落
            current_paragraph = " ".join(sentences[current_start:boundary])
            paragraph_length = len(current_paragraph)
            
            # 长度检查
            if paragraph_length < config["min_paragraph_length"]:
                # 段落太短，尝试跳过这个边界
                continue
            elif paragraph_length > config["max_paragraph_length"]:
                # 段落太长，在中间强制分割
                mid_point = current_start + (boundary - current_start) // 2
                if mid_point > current_start:
                    optimized.append(mid_point)
            
            optimized.append(boundary)
            current_start = boundary
        
        return optimized
    
    def _detect_structure_boundaries(self, sentences: List[str]) -> List[int]:
        """
        检测结构性边界（如标题、列表等）
        
        Args:
            sentences: 句子列表
            
        Returns:
            结构边界列表
        """
        boundaries = []
        
        for i, sentence in enumerate(sentences):
            # 检测可能的标题模式
            if (len(sentence) < 50 and 
                (sentence.startswith(('#', '##', '###')) or  # Markdown标题
                 sentence.endswith('：') or  # 中文冒号结尾
                 sentence.endswith(':') or   # 英文冒号结尾
                 len(sentence.split()) <= 5)):  # 短句可能是标题
                if i > 0:  # 不在开头
                    boundaries.append(i)
            
            # 检测列表项
            if (sentence.strip().startswith(('•', '-', '*', '1.', '2.', '3.')) or
                sentence.strip().startswith(('一、', '二、', '三、', '（一）', '（二）'))):
                if i > 0:
                    boundaries.append(i)
        
        return boundaries
    
    def _build_hierarchical_paragraphs(self, sentences: List[str], 
                                     boundaries: List[int]) -> List[Dict[str, Any]]:
        """
        构建层次化段落结构
        
        Args:
            sentences: 句子列表
            boundaries: 边界列表
            
        Returns:
            层次化段落列表
        """
        if not sentences:
            return []
        
        hierarchical_paragraphs = []
        all_boundaries = [0] + sorted(boundaries) + [len(sentences)]
        
        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]
            
            if start_idx < end_idx:
                paragraph_sentences = sentences[start_idx:end_idx]
                paragraph_text = " ".join(paragraph_sentences)
                paragraph_text = self.text_processor.normalize_text(paragraph_text)
                
                if paragraph_text.strip():
                    # 提取关键短语
                    key_phrases = self.model.extract_key_phrases([paragraph_text], top_k=3)[0]
                    
                    # 判断段落类型
                    paragraph_type = "content"
                    if len(paragraph_text) < 50:
                        paragraph_type = "title"
                    elif any(paragraph_text.startswith(prefix) for prefix in ['•', '-', '*', '1.', '2.']):
                        paragraph_type = "list_item"
                    
                    hierarchical_paragraphs.append({
                        "text": paragraph_text.strip(),
                        "type": paragraph_type,
                        "length": len(paragraph_text),
                        "sentence_count": len(paragraph_sentences),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "key_phrases": key_phrases,
                        "level": 1  # 可以后续扩展为多层次
                    })
        
        return hierarchical_paragraphs
    
    def segment_text_enhanced(self, text: str, 
                            custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        增强版文本分段
        
        Args:
            text: 输入文本
            custom_config: 自定义配置
            
        Returns:
            增强版分段结果
        """
        try:
            # 输入验证
            if not self.text_processor.validate_input(text):
                return {"error": "输入文本无效"}
            
            # 检测文本类型
            text_type, type_info = self.type_detector.detect_text_type(text)
            logger.info(f"检测到文本类型: {text_type.value}")
            
            # 获取类型特定配置
            type_config = self.type_detector.get_segmentation_config(text_type)
            
            # 合并配置
            final_config = {**self.default_config, **type_config}
            if custom_config:
                final_config.update(custom_config)
            
            # 预处理文本
            normalized_text = self.text_processor.normalize_text(text)
            
            # 分句
            sentences = self.text_processor.split_sentences(normalized_text)
            if len(sentences) < 2:
                return {
                    "paragraphs": [{"text": normalized_text, "type": "content", "level": 1}] if normalized_text else [],
                    "text_type": text_type.value,
                    "config_used": final_config,
                    "quality": {"message": "文本太短，无需分段"}
                }
            
            # 编码句子
            embeddings = self.model.encode_texts(sentences)
            
            # 计算多尺度相似度
            multi_scale_similarities = self._calculate_multi_scale_similarities(
                embeddings, scales=[1, 3, 5]
            )
            
            # 综合确定边界
            boundaries = self._combine_multi_scale_boundaries(
                multi_scale_similarities, final_config["threshold"], text_type
            )
            
            # 如果配置要求，检测结构边界
            if final_config.get("use_structure_hints", False):
                structure_boundaries = self._detect_structure_boundaries(sentences)
                boundaries.extend(structure_boundaries)
                boundaries = sorted(list(set(boundaries)))
            
            # 优化边界
            optimized_boundaries = self._optimize_boundaries_by_content(
                sentences, boundaries, final_config
            )
            
            # 构建层次化段落
            hierarchical_paragraphs = self._build_hierarchical_paragraphs(
                sentences, optimized_boundaries
            )
            
            # 评估分段质量
            quality_metrics = self._evaluate_enhanced_quality(hierarchical_paragraphs, final_config)
            
            return {
                "paragraphs": hierarchical_paragraphs,
                "text_type": text_type.value,
                "type_confidence": type_info["confidence"],
                "config_used": final_config,
                "sentence_count": len(sentences),
                "boundary_count": len(optimized_boundaries),
                "multi_scale_info": {scale: len(sims) for scale, sims in multi_scale_similarities.items()},
                "quality": quality_metrics
            }
            
        except Exception as e:
            logger.error(f"增强版文本分段失败: {e}")
            return {"error": f"分段处理错误: {str(e)}"}
    
    def _evaluate_enhanced_quality(self, paragraphs: List[Dict[str, Any]], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估增强版分段质量
        
        Args:
            paragraphs: 层次化段落列表
            config: 分段配置
            
        Returns:
            质量评估指标
        """
        if not paragraphs:
            return {"error": "无段落可评估"}
        
        # 基本统计
        lengths = [p["length"] for p in paragraphs]
        sentence_counts = [p["sentence_count"] for p in paragraphs]
        
        # 类型分布
        type_distribution = {}
        for p in paragraphs:
            p_type = p.get("type", "content")
            type_distribution[p_type] = type_distribution.get(p_type, 0) + 1
        
        # 长度合理性
        too_short = sum(1 for l in lengths if l < config["min_paragraph_length"])
        too_long = sum(1 for l in lengths if l > config["max_paragraph_length"])
        
        # 语义一致性（计算段落间的平均相似度）
        semantic_consistency = 0.0
        if len(paragraphs) > 1:
            try:
                paragraph_texts = [p["text"] for p in paragraphs]
                embeddings = self.model.encode_texts(paragraph_texts)
                similarities = self.model.batch_similarities(embeddings)
                semantic_consistency = np.mean(similarities) if similarities else 0.0
            except Exception:
                semantic_consistency = 0.0
        
        # 结构质量（标题和内容的比例）
        content_ratio = type_distribution.get("content", 0) / len(paragraphs)
        structure_score = 1.0 - abs(content_ratio - 0.8)  # 理想情况下80%是内容
        
        # 综合质量分数
        quality_score = self._calculate_enhanced_quality_score(
            len(paragraphs), np.mean(lengths), too_short, too_long,
            semantic_consistency, structure_score
        )
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_length": round(np.mean(lengths), 2),
            "length_std": round(np.std(lengths), 2),
            "avg_sentence_count": round(np.mean(sentence_counts), 2),
            "too_short_count": too_short,
            "too_long_count": too_long,
            "semantic_consistency": round(semantic_consistency, 3),
            "structure_score": round(structure_score, 3),
            "type_distribution": type_distribution,
            "quality_score": round(quality_score, 3)
        }
    
    def _calculate_enhanced_quality_score(self, count: int, avg_length: float,
                                        too_short: int, too_long: int,
                                        consistency: float, structure: float) -> float:
        """计算增强版质量分数"""
        score = 1.0
        
        # 长度合理性
        if too_short > 0:
            score -= 0.1 * too_short / count
        if too_long > 0:
            score -= 0.1 * too_long / count
        
        # 语义一致性
        score *= (1 + consistency * 0.2)
        
        # 结构合理性
        score *= structure
        
        # 段落数量合理性
        if count < 2:
            score *= 0.8
        elif count > 20:
            score *= 0.9
        
        return max(0.0, min(1.0, score))