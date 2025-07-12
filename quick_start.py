#!/usr/bin/env python3
"""
快速启动脚本 - 使用模拟BERT模型进行演示
"""

import sys
import os
import uvicorn
from unittest.mock import Mock
import torch

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_app():
    """创建使用模拟BERT模型的应用"""
    from src.api.app import app, get_segmenter
    from src.core.semantic_segmenter import SemanticSegmenter
    from src.core.bert_model import BERTModel
    from src.utils.text_processor import TextProcessor
    
    # 创建模拟的BERT模型
    mock_bert = Mock(spec=BERTModel)
    mock_bert.model_name = "mock-bert-model"
    mock_bert.device = "cpu"
    mock_bert.max_length = 512
    mock_bert._is_loaded = True
    
    # 模拟编码功能
    def mock_encode_texts(texts):
        # 返回随机张量模拟BERT输出
        return torch.randn(len(texts), 768)
    
    def mock_calculate_similarity(emb1, emb2):
        # 返回随机相似度
        import random
        return random.uniform(0.3, 0.9)
    
    def mock_batch_similarities(embeddings):
        import random
        return [random.uniform(0.3, 0.9) for _ in range(embeddings.size(0) - 1)]
    
    def mock_get_model_info():
        return {
            "model_name": "mock-bert-model",
            "device": "cpu",
            "is_loaded": True,
            "note": "这是演示模式，使用模拟BERT模型"
        }
    
    mock_bert.encode_texts = mock_encode_texts
    mock_bert.calculate_similarity = mock_calculate_similarity
    mock_bert.batch_similarities = mock_batch_similarities
    mock_bert.get_model_info = mock_get_model_info
    mock_bert.load_model = Mock()
    mock_bert.is_loaded = Mock(return_value=True)
    mock_bert.get_device = Mock(return_value="cpu")
    
    # 创建分段器
    text_processor = TextProcessor()
    segmenter = SemanticSegmenter(mock_bert, text_processor)
    
    # 重写依赖
    app.dependency_overrides[get_segmenter] = lambda: segmenter
    
    return app

def main():
    """主函数"""
    print("=== BERT语义分段服务 - 演示模式 ===")
    print("注意: 使用模拟BERT模型，仅用于演示API功能")
    print()
    
    try:
        # 创建应用
        demo_app = create_mock_app()
        
        print("正在启动演示服务...")
        print("服务地址: http://localhost:8000")
        print("API文档: http://localhost:8000/docs")
        print("健康检查: http://localhost:8000/health")
        print()
        print("按 Ctrl+C 停止服务")
        print()
        
        # 启动服务
        uvicorn.run(
            demo_app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()