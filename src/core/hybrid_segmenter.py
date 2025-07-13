import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum

from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
from src.core.transformer_segmenter import TransformerSemanticSegmenter
from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector, TextType
from src.utils.text_processor import TextProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class SegmentationMethod(Enum):
    """分段方法枚举"""
    ENHANCED = "enhanced"          # 增强版多尺度分段器
    TRANSFORMER = "transformer"    # Transformer²分段器
    HYBRID = "hybrid"             # 混合方法
    AUTO = "auto"                 # 自动选择最佳方法


class HybridSemanticSegmenter:
    """
    混合语义分段器 - 智能选择最佳分段方法
    """
    
    def __init__(self, 
                 sentence_model: Optional[SentenceTransformerModel] = None,
                 text_processor: Optional[TextProcessor] = None,
                 type_detector: Optional[TextTypeDetector] = None,
                 device: str = "auto"):
        """
        初始化混合分段器
        
        Args:
            sentence_model: Sentence Transformer模型实例
            text_processor: 文本处理器实例
            type_detector: 文本类型检测器实例
            device: 计算设备
        """
        self.sentence_model = sentence_model or SentenceTransformerModel(device=device)
        self.text_processor = text_processor or TextProcessor()
        self.type_detector = type_detector or TextTypeDetector()
        
        # 初始化各种分段器
        self.enhanced_segmenter = EnhancedSemanticSegmenter(
            self.sentence_model, self.text_processor, self.type_detector
        )
        
        self.transformer_segmenter = TransformerSemanticSegmenter(
            self.sentence_model, self.text_processor, self.type_detector, device
        )
        
        # 方法选择策略配置
        self.method_selection_config = {
            # 根据文本类型推荐方法
            TextType.TECHNICAL: SegmentationMethod.ENHANCED,     # 技术文档用增强版
            TextType.NOVEL: SegmentationMethod.TRANSFORMER,      # 小说用Transformer
            TextType.ACADEMIC: SegmentationMethod.ENHANCED,      # 学术文档用增强版
            TextType.NEWS: SegmentationMethod.ENHANCED,          # 新闻用增强版
            TextType.DIALOGUE: SegmentationMethod.TRANSFORMER,   # 对话用Transformer
            TextType.MIXED: SegmentationMethod.HYBRID,           # 混合文本用混合方法
            TextType.UNKNOWN: SegmentationMethod.ENHANCED        # 未知类型用增强版
        }
        
        # 文本长度阈值配置
        self.length_thresholds = {
            "short": 500,       # 短文本阈值
            "medium": 2000,     # 中等文本阈值
            "long": 5000        # 长文本阈值
        }
        
        logger.info("混合语义分段器初始化完成")
    
    def _select_optimal_method(self, text: str, text_type: TextType, 
                             user_preference: Optional[SegmentationMethod] = None) -> SegmentationMethod:
        """
        选择最优分段方法
        
        Args:
            text: 输入文本
            text_type: 文本类型
            user_preference: 用户偏好方法
            
        Returns:
            选择的分段方法
        """
        # 用户指定方法优先
        if user_preference and user_preference != SegmentationMethod.AUTO:
            return user_preference
        
        text_length = len(text)
        
        # 根据文本长度调整策略
        if text_length < self.length_thresholds["short"]:
            # 短文本使用增强版（更快）
            return SegmentationMethod.ENHANCED
        elif text_length > self.length_thresholds["long"]:
            # 长文本使用增强版（更稳定）
            return SegmentationMethod.ENHANCED
        
        # 中等长度文本根据类型选择
        recommended_method = self.method_selection_config.get(text_type, SegmentationMethod.ENHANCED)
        
        logger.debug(f"为文本类型 {text_type.value} (长度: {text_length}) 选择方法: {recommended_method.value}")
        return recommended_method
    
    def _merge_results(self, enhanced_result: Dict[str, Any], 
                      transformer_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两种方法的结果
        
        Args:
            enhanced_result: 增强版分段结果
            transformer_result: Transformer分段结果
            
        Returns:
            合并后的结果
        """
        # 比较质量分数选择更好的结果
        enhanced_quality = enhanced_result.get("quality", {}).get("quality_score", 0.0)
        transformer_quality = transformer_result.get("quality", {}).get("quality_score", 0.0)
        
        if enhanced_quality >= transformer_quality:
            best_result = enhanced_result.copy()
            best_method = "enhanced"
            alternative_quality = transformer_quality
        else:
            best_result = transformer_result.copy()
            best_method = "transformer"
            alternative_quality = enhanced_quality
        
        # 添加比较信息
        best_result["method"] = "hybrid"
        best_result["selected_method"] = best_method
        best_result["quality_comparison"] = {
            "enhanced_score": enhanced_quality,
            "transformer_score": transformer_quality,
            "selected_score": best_result["quality"]["quality_score"],
            "alternative_score": alternative_quality,
            "improvement": best_result["quality"]["quality_score"] - alternative_quality
        }
        
        return best_result
    
    def segment_text(self, text: str, 
                    method: Union[str, SegmentationMethod] = SegmentationMethod.AUTO,
                    custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        统一的文本分段接口
        
        Args:
            text: 输入文本
            method: 分段方法
            custom_config: 自定义配置
            
        Returns:
            分段结果
        """
        start_time = time.time()
        
        try:
            # 类型转换
            if isinstance(method, str):
                method = SegmentationMethod(method)
            
            # 文本类型检测
            text_type, type_info = self.type_detector.detect_text_type(text)
            
            # 选择最优方法
            selected_method = self._select_optimal_method(text, text_type, method)
            
            logger.info(f"使用分段方法: {selected_method.value} (文本类型: {text_type.value})")
            
            # 执行分段
            if selected_method == SegmentationMethod.ENHANCED:
                result = self.enhanced_segmenter.segment_text_enhanced(text, custom_config)
                result["method"] = "enhanced"
                
            elif selected_method == SegmentationMethod.TRANSFORMER:
                result = self.transformer_segmenter.segment_text_transformer(text, custom_config)
                result["method"] = "transformer"
                
            elif selected_method == SegmentationMethod.HYBRID:
                # 同时运行两种方法并合并结果
                enhanced_result = self.enhanced_segmenter.segment_text_enhanced(text, custom_config)
                transformer_result = self.transformer_segmenter.segment_text_transformer(text, custom_config)
                result = self._merge_results(enhanced_result, transformer_result)
                
            else:  # AUTO
                # 根据策略选择方法
                optimal_method = self._select_optimal_method(text, text_type)
                return self.segment_text(text, optimal_method, custom_config)
            
            # 添加统一信息
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 3)
            result["selected_method"] = selected_method.value
            
            # 错误处理
            if "error" in result:
                return result
            
            # 成功结果验证
            if "paragraphs" not in result or not isinstance(result["paragraphs"], list):
                return {"error": "分段结果格式错误"}
            
            return result
            
        except Exception as e:
            logger.error(f"混合分段失败: {e}")
            return {"error": f"分段处理错误: {str(e)}"}
    
    def compare_methods(self, text: str, 
                       custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        比较不同分段方法的效果
        
        Args:
            text: 输入文本
            custom_config: 自定义配置
            
        Returns:
            方法比较结果
        """
        start_time = time.time()
        
        try:
            results = {}
            
            # 增强版分段
            enhanced_start = time.time()
            enhanced_result = self.enhanced_segmenter.segment_text_enhanced(text, custom_config)
            enhanced_time = time.time() - enhanced_start
            
            if "error" not in enhanced_result:
                results["enhanced"] = {
                    "result": enhanced_result,
                    "processing_time": enhanced_time,
                    "paragraph_count": len(enhanced_result.get("paragraphs", [])),
                    "quality_score": enhanced_result.get("quality", {}).get("quality_score", 0.0)
                }
            
            # Transformer分段
            transformer_start = time.time()
            transformer_result = self.transformer_segmenter.segment_text_transformer(text, custom_config)
            transformer_time = time.time() - transformer_start
            
            if "error" not in transformer_result:
                results["transformer"] = {
                    "result": transformer_result,
                    "processing_time": transformer_time,
                    "paragraph_count": len(transformer_result.get("paragraphs", [])),
                    "quality_score": transformer_result.get("quality", {}).get("quality_score", 0.0)
                }
            
            # 比较分析
            comparison = {
                "total_processing_time": time.time() - start_time,
                "methods_compared": list(results.keys()),
                "text_length": len(text),
                "sentence_count": len(self.text_processor.split_sentences(text))
            }
            
            if len(results) > 1:
                # 质量比较
                quality_scores = {method: data["quality_score"] for method, data in results.items()}
                best_method = max(quality_scores, key=quality_scores.get)
                
                comparison.update({
                    "best_method": best_method,
                    "quality_difference": max(quality_scores.values()) - min(quality_scores.values()),
                    "speed_comparison": {method: data["processing_time"] for method, data in results.items()},
                    "paragraph_count_comparison": {method: data["paragraph_count"] for method, data in results.items()}
                })
            
            return {
                "comparison": comparison,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"方法比较失败: {e}")
            return {"error": f"比较处理错误: {str(e)}"}
    
    def get_method_recommendation(self, text: str) -> Dict[str, Any]:
        """
        获取方法推荐
        
        Args:
            text: 输入文本
            
        Returns:
            方法推荐信息
        """
        try:
            text_type, type_info = self.type_detector.detect_text_type(text)
            text_length = len(text)
            
            recommended_method = self._select_optimal_method(text, text_type)
            
            # 推荐原因
            reasons = []
            
            if text_length < self.length_thresholds["short"]:
                reasons.append("文本较短，增强版方法更快速")
            elif text_length > self.length_thresholds["long"]:
                reasons.append("文本较长，增强版方法更稳定")
            
            if text_type == TextType.MIXED:
                reasons.append("混合文本类型，建议使用混合方法")
            elif text_type in [TextType.NOVEL, TextType.DIALOGUE]:
                reasons.append("文学/对话类型，Transformer方法理解更好")
            else:
                reasons.append("技术/学术类型，增强版方法更精准")
            
            return {
                "recommended_method": recommended_method.value,
                "text_type": text_type.value,
                "text_length": text_length,
                "confidence": type_info["confidence"],
                "reasons": reasons,
                "alternative_methods": [
                    method.value for method in SegmentationMethod 
                    if method != recommended_method and method != SegmentationMethod.AUTO
                ]
            }
            
        except Exception as e:
            logger.error(f"获取推荐失败: {e}")
            return {"error": f"推荐处理错误: {str(e)}"}