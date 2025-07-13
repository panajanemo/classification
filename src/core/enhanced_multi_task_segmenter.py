import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import time

from src.core.multi_task_framework import (
    MultiTaskModel, MultiTaskTrainer, MultiTaskDataset, 
    TrainingExample, create_synthetic_data
)
from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector, TextType
from src.utils.text_processor import TextProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class EnhancedMultiTaskSegmenter:
    """
    增强版多任务语义分段器
    结合文本分类和语义分段的联合学习
    """
    
    def __init__(self, 
                 sentence_model: Optional[SentenceTransformerModel] = None,
                 text_processor: Optional[TextProcessor] = None,
                 type_detector: Optional[TextTypeDetector] = None,
                 device: str = "auto",
                 model_path: Optional[str] = None):
        """
        初始化增强版多任务分段器
        
        Args:
            sentence_model: Sentence Transformer模型实例
            text_processor: 文本处理器实例
            type_detector: 文本类型检测器实例
            device: 计算设备
            model_path: 预训练模型路径
        """
        self.sentence_model = sentence_model or SentenceTransformerModel(device=device)
        self.text_processor = text_processor or TextProcessor()
        self.type_detector = type_detector or TextTypeDetector()
        
        # 确保sentence model已加载
        if not self.sentence_model.is_loaded():
            self.sentence_model.load_model()
        
        # 设备配置
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化多任务模型
        embedding_dim = self.sentence_model.model.get_sentence_embedding_dimension()
        self.multi_task_model = MultiTaskModel(
            embedding_dim=embedding_dim,
            num_text_types=7,  # 支持7种文本类型
            num_heads=6,
            num_layers=4,
            dropout=0.1,
            shared_layers=2
        ).to(self.device)
        
        # 加载预训练模型或使用随机初始化
        self.is_trained = False
        if model_path and os.path.exists(model_path):
            self._load_pretrained_model(model_path)
        else:
            logger.info("使用随机初始化的多任务模型")
        
        # 配置参数
        self.default_config = {
            "text_type_threshold": 0.7,    # 文本类型置信度阈值
            "boundary_threshold": 0.5,     # 边界检测阈值
            "consistency_threshold": 0.3,  # 一致性阈值
            "use_joint_prediction": True,  # 是否使用联合预测
            "fallback_to_rule": True,      # 是否回退到规则方法
            "min_paragraph_length": settings.min_paragraph_length,
            "max_paragraph_length": settings.max_paragraph_length
        }
        
        # 文本类型映射
        self.id_to_type = {
            0: TextType.TECHNICAL,
            1: TextType.NOVEL,
            2: TextType.ACADEMIC,
            3: TextType.NEWS,
            4: TextType.DIALOGUE,
            5: TextType.MIXED,
            6: TextType.UNKNOWN
        }
        
        logger.info(f"增强版多任务分段器初始化完成，设备: {self.device}")
    
    def _load_pretrained_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.multi_task_model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True
            logger.info(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}，使用随机初始化")
    
    def _prepare_input(self, sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备模型输入"""
        # 编码句子
        embeddings = self.sentence_model.encode_texts(sentences)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        embeddings = embeddings.to(self.device).unsqueeze(0)  # 添加batch维度
        
        # 创建注意力掩码
        seq_len = embeddings.size(1)
        attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)
        
        return embeddings, attention_mask
    
    def _joint_text_type_prediction(self, predictions: Dict[str, torch.Tensor]) -> Tuple[TextType, float]:
        """联合文本类型预测"""
        logits = predictions['text_type_logits'][0]  # 移除batch维度
        probabilities = torch.softmax(logits, dim=0)
        
        predicted_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_id].item()
        
        predicted_type = self.id_to_type.get(predicted_id, TextType.UNKNOWN)
        return predicted_type, confidence
    
    def _extract_boundaries_from_predictions(self, predictions: Dict[str, torch.Tensor],
                                           config: Dict[str, Any]) -> List[int]:
        """从预测结果中提取边界"""
        boundary_preds = predictions['boundary_predictions'][0].cpu().numpy()  # 移除batch维度
        consistency_preds = predictions['consistency_predictions'][0].cpu().numpy()
        
        boundaries = []
        boundary_threshold = config["boundary_threshold"]
        consistency_threshold = config["consistency_threshold"]
        
        for i, (b_score, c_score) in enumerate(zip(boundary_preds, consistency_preds)):
            # 联合判断：边界分数高且一致性分数低
            if b_score > boundary_threshold and c_score < consistency_threshold:
                boundaries.append(i + 1)  # +1因为这是句子i和i+1之间的边界
        
        return boundaries
    
    def _fallback_to_enhanced_method(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """回退到增强版方法"""
        from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
        
        enhanced_segmenter = EnhancedSemanticSegmenter(
            self.sentence_model, self.text_processor, self.type_detector
        )
        
        result = enhanced_segmenter.segment_text_enhanced(text, config)
        result["method"] = "enhanced_fallback"
        result["note"] = "多任务模型未训练，使用增强版方法"
        
        return result
    
    def segment_text_multi_task(self, text: str, 
                              custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        使用多任务学习进行文本分段
        
        Args:
            text: 输入文本
            custom_config: 自定义配置
            
        Returns:
            分段结果
        """
        start_time = time.time()
        
        try:
            # 输入验证
            if not self.text_processor.validate_input(text):
                return {"error": "输入文本无效"}
            
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
                    "text_type": "unknown",
                    "method": "multi_task",
                    "config_used": final_config,
                    "quality": {"message": "文本太短，无需分段"}
                }
            
            # 如果模型未训练且配置允许回退
            if not self.is_trained and final_config.get("fallback_to_rule", True):
                fallback_result = self._fallback_to_enhanced_method(text, final_config)
                fallback_result["is_trained"] = False
                fallback_result["processing_time"] = time.time() - start_time
                return fallback_result
            
            # 准备输入
            embeddings, attention_mask = self._prepare_input(sentences)
            
            # 多任务预测
            self.multi_task_model.eval()
            with torch.no_grad():
                predictions = self.multi_task_model(embeddings, attention_mask, return_features=True)
            
            # 联合文本类型预测
            if final_config.get("use_joint_prediction", True):
                predicted_type, type_confidence = self._joint_text_type_prediction(predictions)
            else:
                # 回退到传统类型检测
                predicted_type, type_info = self.type_detector.detect_text_type(text)
                type_confidence = type_info["confidence"]
            
            # 提取分段边界
            boundaries = self._extract_boundaries_from_predictions(predictions, final_config)
            
            # 构建段落
            paragraphs = []
            all_boundaries = [0] + sorted(boundaries) + [len(sentences)]
            
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
                            "method": "multi_task"
                        })
            
            # 评估质量
            quality_metrics = self._evaluate_multi_task_quality(
                paragraphs, predictions, sentences
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            return {
                "paragraphs": paragraphs,
                "text_type": predicted_type.value if hasattr(predicted_type, 'value') else str(predicted_type),
                "type_confidence": type_confidence,
                "method": "multi_task",
                "is_trained": self.is_trained,
                "config_used": final_config,
                "sentence_count": len(sentences),
                "boundary_count": len(boundaries),
                "multi_task_predictions": {
                    "boundary_scores": predictions['boundary_predictions'][0].cpu().numpy().tolist(),
                    "consistency_scores": predictions['consistency_predictions'][0].cpu().numpy().tolist(),
                    "text_type_probabilities": torch.softmax(predictions['text_type_logits'][0], dim=0).cpu().numpy().tolist()
                },
                "quality": quality_metrics,
                "processing_time": round(processing_time, 3)
            }
            
        except Exception as e:
            logger.error(f"多任务分段失败: {e}")
            return {"error": f"分段处理错误: {str(e)}"}
    
    def _evaluate_multi_task_quality(self, paragraphs: List[Dict[str, Any]], 
                                   predictions: Dict[str, torch.Tensor],
                                   sentences: List[str]) -> Dict[str, Any]:
        """评估多任务分段质量"""
        if not paragraphs:
            return {"error": "无段落可评估"}
        
        # 基本统计
        lengths = [p["length"] for p in paragraphs]
        sentence_counts = [p["sentence_count"] for p in paragraphs]
        
        # 多任务特定指标
        boundary_preds = predictions['boundary_predictions'][0].cpu().numpy()
        consistency_preds = predictions['consistency_predictions'][0].cpu().numpy()
        
        avg_boundary_score = float(np.mean(boundary_preds)) if len(boundary_preds) > 0 else 0.0
        avg_consistency_score = float(np.mean(consistency_preds)) if len(consistency_preds) > 0 else 0.0
        
        # 预测置信度
        text_type_probs = torch.softmax(predictions['text_type_logits'][0], dim=0).cpu().numpy()
        max_type_confidence = float(np.max(text_type_probs))
        
        # 综合质量分数
        quality_score = self._calculate_multi_task_quality_score(
            len(paragraphs), np.mean(lengths), avg_boundary_score, 
            avg_consistency_score, max_type_confidence
        )
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_length": round(np.mean(lengths), 2),
            "length_std": round(np.std(lengths), 2),
            "avg_sentence_count": round(np.mean(sentence_counts), 2),
            "avg_boundary_score": round(avg_boundary_score, 3),
            "avg_consistency_score": round(avg_consistency_score, 3),
            "text_type_confidence": round(max_type_confidence, 3),
            "prediction_entropy": round(float(-np.sum(text_type_probs * np.log(text_type_probs + 1e-9))), 3),
            "quality_score": round(quality_score, 3),
            "method": "multi_task"
        }
    
    def _calculate_multi_task_quality_score(self, count: int, avg_length: float,
                                          boundary_score: float, consistency_score: float,
                                          type_confidence: float) -> float:
        """计算多任务质量分数"""
        score = 1.0
        
        # 文本类型预测置信度奖励
        score *= (1 + type_confidence * 0.2)
        
        # 边界预测质量
        if 0.3 <= boundary_score <= 0.7:  # 适中的边界分数最好
            score *= 1.1
        
        # 一致性预测质量
        if 0.4 <= consistency_score <= 0.8:  # 适中的一致性最好
            score *= 1.1
        
        # 段落数量合理性
        if 2 <= count <= 8:
            score *= 1.2
        elif count == 1:
            score *= 0.8
        elif count > 15:
            score *= 0.9
        
        # 平均长度合理性
        if 100 <= avg_length <= 400:
            score *= 1.1
        elif avg_length < 50:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def train_model(self, training_data: Optional[List[TrainingExample]] = None,
                   epochs: int = 10, batch_size: int = 8,
                   learning_rate: float = 1e-4, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        训练多任务模型
        
        Args:
            training_data: 训练数据，如果为None则使用合成数据
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            save_path: 模型保存路径
            
        Returns:
            训练统计信息
        """
        # 准备训练数据
        if training_data is None:
            logger.info("使用合成数据进行训练")
            training_data = create_synthetic_data(num_samples=200)
        
        # 创建数据集和数据加载器
        dataset = MultiTaskDataset(training_data, self.sentence_model)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # 创建训练器
        trainer = MultiTaskTrainer(
            self.multi_task_model, 
            self.sentence_model,
            device=str(self.device),
            learning_rate=learning_rate
        )
        
        logger.info(f"开始训练，{epochs}轮，批大小{batch_size}")
        
        # 训练循环
        training_history = []
        for epoch in range(epochs):
            epoch_losses = trainer.train_epoch(dataloader)
            training_history.append(epoch_losses)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Total Loss: {epoch_losses['total']:.4f}, "
                       f"Classification: {epoch_losses['classification']:.4f}, "
                       f"Boundary: {epoch_losses['boundary']:.4f}, "
                       f"Consistency: {epoch_losses['consistency']:.4f}")
        
        # 标记为已训练
        self.is_trained = True
        
        # 保存模型
        if save_path:
            trainer.save_model(save_path)
            logger.info(f"模型已保存到: {save_path}")
        
        return {
            "training_completed": True,
            "epochs": epochs,
            "final_loss": training_history[-1] if training_history else None,
            "training_history": training_history,
            "model_saved": save_path is not None
        }
    
    def compare_with_other_methods(self, text: str) -> Dict[str, Any]:
        """与其他方法进行比较"""
        try:
            from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
            from src.core.transformer_segmenter import TransformerSemanticSegmenter
            
            results = {}
            
            # 多任务方法
            start_time = time.time()
            multi_task_result = self.segment_text_multi_task(text)
            multi_task_time = time.time() - start_time
            
            if "error" not in multi_task_result:
                results["multi_task"] = {
                    "result": multi_task_result,
                    "processing_time": multi_task_time,
                    "paragraph_count": len(multi_task_result.get("paragraphs", [])),
                    "quality_score": multi_task_result.get("quality", {}).get("quality_score", 0.0)
                }
            
            # 增强版方法
            enhanced_segmenter = EnhancedSemanticSegmenter(
                self.sentence_model, self.text_processor, self.type_detector
            )
            start_time = time.time()
            enhanced_result = enhanced_segmenter.segment_text_enhanced(text)
            enhanced_time = time.time() - start_time
            
            if "error" not in enhanced_result:
                results["enhanced"] = {
                    "result": enhanced_result,
                    "processing_time": enhanced_time,
                    "paragraph_count": len(enhanced_result.get("paragraphs", [])),
                    "quality_score": enhanced_result.get("quality", {}).get("quality_score", 0.0)
                }
            
            # Transformer方法
            transformer_segmenter = TransformerSemanticSegmenter(
                self.sentence_model, self.text_processor, self.type_detector, str(self.device)
            )
            start_time = time.time()
            transformer_result = transformer_segmenter.segment_text_transformer(text)
            transformer_time = time.time() - start_time
            
            if "error" not in transformer_result:
                results["transformer"] = {
                    "result": transformer_result,
                    "processing_time": transformer_time,
                    "paragraph_count": len(transformer_result.get("paragraphs", [])),
                    "quality_score": transformer_result.get("quality", {}).get("quality_score", 0.0)
                }
            
            # 比较分析
            if len(results) > 1:
                quality_scores = {method: data["quality_score"] for method, data in results.items()}
                best_method = max(quality_scores, key=quality_scores.get) if quality_scores else None
                
                comparison = {
                    "best_method": best_method,
                    "quality_comparison": quality_scores,
                    "speed_comparison": {method: data["processing_time"] for method, data in results.items()},
                    "paragraph_count_comparison": {method: data["paragraph_count"] for method, data in results.items()}
                }
                
                return {
                    "comparison": comparison,
                    "results": results,
                    "text_length": len(text),
                    "is_multi_task_trained": self.is_trained
                }
            else:
                return {"error": "无法进行比较，所有方法都失败了"}
                
        except Exception as e:
            logger.error(f"方法比较失败: {e}")
            return {"error": f"比较处理错误: {str(e)}"}