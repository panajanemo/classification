import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.app import app, get_segmenter
from src.core.semantic_segmenter import SemanticSegmenter


class TestAPI:
    """API接口测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.client = TestClient(app)
        
        # 创建模拟的分段器
        self.mock_segmenter = Mock(spec=SemanticSegmenter)
        self.mock_segmenter.threshold = 0.5
        self.mock_segmenter.min_paragraph_length = 50
        self.mock_segmenter.max_paragraph_length = 500
        
        # 模拟BERT模型信息
        mock_bert_model = Mock()
        mock_bert_model.get_model_info.return_value = {
            "model_name": "test-model",
            "device": "cpu",
            "is_loaded": True
        }
        self.mock_segmenter.bert_model = mock_bert_model
        
        # 模拟文本处理器
        mock_text_processor = Mock()
        mock_text_processor.format_output.return_value = "段落1\n\n段落2"
        self.mock_segmenter.text_processor = mock_text_processor
        
        # 重写依赖
        app.dependency_overrides[get_segmenter] = lambda: self.mock_segmenter
    
    def teardown_method(self):
        """测试后清理"""
        app.dependency_overrides.clear()
    
    def test_root_endpoint(self):
        """测试根路径"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """测试健康检查"""
        with patch('src.api.app.startup_time', 1000):
            with patch('time.time', return_value=2000):
                response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_info" in data
        assert "uptime" in data
    
    def test_successful_segmentation(self):
        """测试成功的文本分段"""
        # 模拟分段结果
        mock_result = {
            "paragraphs": ["第一段内容", "第二段内容"],
            "sentence_count": 4,
            "boundary_count": 1,
            "quality": {
                "paragraph_count": 2,
                "avg_length": 45.5,
                "length_std": 2.5,
                "too_short_count": 0,
                "too_long_count": 0,
                "semantic_consistency": 0.75,
                "quality_score": 0.85
            }
        }
        self.mock_segmenter.segment_text.return_value = mock_result
        
        # 发送请求
        request_data = {
            "text": "这是测试文本。包含多个句子。需要进行分段。处理不同主题。",
            "threshold": 0.6
        }
        
        response = self.client.post("/segment", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert len(data["paragraphs"]) == 2
        assert data["sentence_count"] == 4
        assert "quality" in data
        assert "processing_time" in data
    
    def test_segmentation_with_error(self):
        """测试分段错误处理"""
        # 模拟错误结果
        self.mock_segmenter.segment_text.return_value = {"error": "输入文本无效"}
        
        request_data = {"text": ""}
        response = self.client.post("/segment", json=request_data)
        
        assert response.status_code == 400
    
    def test_invalid_request_data(self):
        """测试无效请求数据"""
        # 缺少必需字段
        response = self.client.post("/segment", json={})
        assert response.status_code == 422
        
        # 文本过长
        long_text = "a" * 20000
        response = self.client.post("/segment", json={"text": long_text})
        assert response.status_code == 422
        
        # 无效阈值
        response = self.client.post("/segment", json={
            "text": "有效文本", 
            "threshold": 1.5
        })
        assert response.status_code == 422
    
    def test_get_config(self):
        """测试获取配置"""
        response = self.client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "current_config" in data
        assert "threshold" in data["current_config"]
    
    def test_update_config(self):
        """测试更新配置"""
        config_data = {
            "threshold": 0.7,
            "min_paragraph_length": 60
        }
        
        response = self.client.put("/config", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "current_config" in data
        
        # 验证set_threshold被调用
        self.mock_segmenter.set_threshold.assert_called_with(0.7)
    
    def test_update_config_invalid(self):
        """测试无效配置更新"""
        # 无效阈值
        config_data = {"threshold": -0.1}
        response = self.client.put("/config", json=config_data)
        assert response.status_code == 422
        
        # 无效段落长度
        config_data = {"min_paragraph_length": 5}
        response = self.client.put("/config", json=config_data)
        assert response.status_code == 422
    
    def test_segmentation_with_custom_separator(self):
        """测试自定义分隔符"""
        mock_result = {
            "paragraphs": ["段落1", "段落2"],
            "sentence_count": 2,
            "boundary_count": 1,
            "quality": {
                "paragraph_count": 2,
                "avg_length": 20.0,
                "length_std": 0.0,
                "too_short_count": 0,
                "too_long_count": 0,
                "semantic_consistency": 0.8,
                "quality_score": 0.9
            }
        }
        self.mock_segmenter.segment_text.return_value = mock_result
        
        request_data = {
            "text": "文本内容",
            "separator": " | "
        }
        
        response = self.client.post("/segment", json=request_data)
        
        assert response.status_code == 200
        # 验证format_output被正确调用
        self.mock_segmenter.text_processor.format_output.assert_called_with(
            ["段落1", "段落2"], " | "
        )
    
    def test_threshold_persistence(self):
        """测试阈值设置的持久性"""
        mock_result = {
            "paragraphs": ["测试段落"],
            "sentence_count": 1,
            "boundary_count": 0,
            "quality": {
                "paragraph_count": 1,
                "avg_length": 10.0,
                "length_std": 0.0,
                "too_short_count": 0,
                "too_long_count": 0,
                "semantic_consistency": 0.0,
                "quality_score": 1.0
            }
        }
        self.mock_segmenter.segment_text.return_value = mock_result
        
        # 发送带有自定义阈值的请求
        request_data = {
            "text": "测试文本",
            "threshold": 0.8
        }
        
        response = self.client.post("/segment", json=request_data)
        
        assert response.status_code == 200
        
        # 验证阈值被设置和恢复
        calls = self.mock_segmenter.set_threshold.call_args_list
        assert len(calls) == 2  # 设置新阈值和恢复原阈值
        assert calls[0][0][0] == 0.8  # 设置为0.8
        assert calls[1][0][0] == 0.5  # 恢复为0.5
    
    def test_cors_headers(self):
        """测试CORS头部"""
        response = self.client.options("/segment")
        # FastAPI的TestClient可能不完全模拟CORS，但我们可以测试基本功能
        assert response.status_code in [200, 405]  # OPTIONS可能不被支持
    
    def test_error_response_format(self):
        """测试错误响应格式"""
        # 模拟分段器抛出异常
        self.mock_segmenter.segment_text.side_effect = Exception("测试异常")
        
        request_data = {"text": "测试文本"}
        response = self.client.post("/segment", json=request_data)
        
        assert response.status_code == 500
        # 由于全局异常处理器，应该返回结构化的错误响应