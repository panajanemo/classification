import pytest
import torch
from unittest.mock import Mock, patch
from src.core.semantic_segmenter import SemanticSegmenter
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor


class TestSemanticSegmentation:
    """语义分段集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 使用模拟的BERT模型避免实际模型加载
        self.mock_bert = Mock(spec=BERTModel)
        self.text_processor = TextProcessor()
        self.segmenter = SemanticSegmenter(self.mock_bert, self.text_processor)
    
    def test_segmenter_initialization(self):
        """测试分段器初始化"""
        assert self.segmenter.bert_model is not None
        assert self.segmenter.text_processor is not None
        assert self.segmenter.threshold > 0
    
    def test_continuation_probabilities_calculation(self):
        """测试续写概率计算"""
        sentences = ["第一句话。", "第二句话。", "第三句话。"]
        
        # 模拟BERT编码结果
        mock_embeddings = torch.randn(3, 768)  # 假设768维embedding
        self.mock_bert.encode_texts.return_value = mock_embeddings
        self.mock_bert.calculate_similarity.side_effect = [0.8, 0.3]  # 模拟相似度
        
        probs = self.segmenter.calculate_continuation_probabilities(sentences)
        
        assert len(probs) == 2  # 3个句子应该有2个续写概率
        assert all(0 <= p <= 1 for p in probs)  # 概率值在0-1之间
        self.mock_bert.encode_texts.assert_called_once_with(sentences)
    
    def test_boundary_identification(self):
        """测试段落边界识别"""
        sentences = ["句子1", "句子2", "句子3", "句子4"]
        probabilities = [0.8, 0.3, 0.7]  # 第二个位置概率低，应该分段
        
        # 设置较高阈值确保在第二个位置分段
        self.segmenter.threshold = 0.5
        
        boundaries = self.segmenter.identify_paragraph_boundaries(sentences, probabilities)
        
        assert 2 in boundaries  # 应该在第2个位置分段
    
    def test_paragraph_building(self):
        """测试段落构建"""
        sentences = ["第一句。", "第二句。", "第三句。", "第四句。"]
        boundaries = [2]  # 在第2个位置分段
        
        paragraphs = self.segmenter.build_paragraphs(sentences, boundaries)
        
        assert len(paragraphs) == 2
        assert "第一句" in paragraphs[0]
        assert "第二句" in paragraphs[0]
        assert "第三句" in paragraphs[1]
        assert "第四句" in paragraphs[1]
    
    def test_segmentation_quality_evaluation(self):
        """测试分段质量评估"""
        paragraphs = [
            "这是第一段，有合适的长度。",
            "这是第二段，也有合适的长度。"
        ]
        
        # 模拟质量评估
        mock_embeddings = torch.randn(2, 768)
        self.mock_bert.encode_texts.return_value = mock_embeddings
        self.mock_bert.batch_similarities.return_value = [0.6]
        
        quality = self.segmenter.evaluate_segmentation_quality(paragraphs)
        
        assert "paragraph_count" in quality
        assert "avg_length" in quality
        assert "quality_score" in quality
        assert quality["paragraph_count"] == 2
    
    def test_empty_text_handling(self):
        """测试空文本处理"""
        result = self.segmenter.segment_text("")
        assert "error" in result
        
        result = self.segmenter.segment_text("   ")
        assert "error" in result
    
    def test_single_sentence_handling(self):
        """测试单句文本处理"""
        text = "这是唯一的一句话。"
        result = self.segmenter.segment_text(text)
        
        # 单句应该不需要分段
        assert "paragraphs" in result
        assert len(result["paragraphs"]) == 1
        assert "quality" in result
    
    @patch('src.core.semantic_segmenter.SemanticSegmenter.calculate_continuation_probabilities')
    def test_full_segmentation_workflow(self, mock_calc_probs):
        """测试完整分段流程"""
        # 准备测试数据
        text = "第一句话。第二句话，内容相关。第三句话，不同主题。第四句话，回到相关主题。"
        mock_calc_probs.return_value = [0.8, 0.2, 0.7]  # 在第二个位置分段
        
        # 执行分段
        result = self.segmenter.segment_text(text)
        
        # 验证结果
        assert "paragraphs" in result
        assert "sentence_count" in result
        assert "quality" in result
        assert result["sentence_count"] > 0
        assert len(result["paragraphs"]) > 0
    
    def test_threshold_setting(self):
        """测试阈值设置"""
        original_threshold = self.segmenter.threshold
        
        # 测试有效阈值
        self.segmenter.set_threshold(0.7)
        assert self.segmenter.threshold == 0.7
        
        # 测试无效阈值
        with pytest.raises(ValueError):
            self.segmenter.set_threshold(-0.1)
        
        with pytest.raises(ValueError):
            self.segmenter.set_threshold(1.5)
        
        # 恢复原始阈值
        self.segmenter.set_threshold(original_threshold)
    
    def test_length_optimization(self):
        """测试长度优化"""
        # 测试过短段落合并
        sentences = ["短句1。", "短句2。", "正常长度的句子内容。", "另一个正常长度的句子。"]
        boundaries = [1, 2, 3]  # 每句都分段
        
        optimized = self.segmenter._optimize_boundaries_by_length(sentences, boundaries)
        
        # 优化后应该减少边界数量
        assert len(optimized) <= len(boundaries)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟BERT模型错误
        self.mock_bert.encode_texts.side_effect = Exception("模型错误")
        
        result = self.segmenter.segment_text("测试文本，应该处理错误。")
        assert "error" in result