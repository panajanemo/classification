#!/usr/bin/env python3
"""
快速测试脚本 - 验证服务是否正常工作
"""

import sys
import os
import time

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_text_processor():
    """测试文本处理器"""
    print("1. 测试文本处理器...")
    try:
        from src.utils.text_processor import TextProcessor
        processor = TextProcessor()
        
        text = "这是第一句话。这是第二句话！这是第三句话？"
        sentences = processor.split_sentences(text)
        print(f"   分句成功: {len(sentences)} 句")
        for i, sentence in enumerate(sentences, 1):
            print(f"     句子{i}: {sentence}")
        return True
    except Exception as e:
        print(f"   失败: {e}")
        return False

def test_without_model():
    """测试不加载BERT模型的核心功能"""
    print("2. 测试核心功能(不加载模型)...")
    try:
        from src.core.semantic_segmenter import SemanticSegmenter
        from src.utils.text_processor import TextProcessor
        from unittest.mock import Mock
        
        # 使用模拟的BERT模型
        mock_bert = Mock()
        text_processor = TextProcessor()
        segmenter = SemanticSegmenter(mock_bert, text_processor)
        
        print("   语义分段器初始化成功")
        print(f"   默认阈值: {segmenter.threshold}")
        
        # 测试阈值设置
        segmenter.set_threshold(0.7)
        print(f"   阈值更新为: {segmenter.threshold}")
        
        return True
    except Exception as e:
        print(f"   失败: {e}")
        return False

def test_api_models():
    """测试API模型"""
    print("3. 测试API模型...")
    try:
        from src.api.models import SegmentationRequest, SegmentationResponse
        
        # 测试请求模型
        request = SegmentationRequest(
            text="测试文本内容",
            threshold=0.6
        )
        print(f"   请求模型创建成功: {request.text[:10]}...")
        
        return True
    except Exception as e:
        print(f"   失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 语义分段服务快速测试 ===\n")
    
    tests = [
        test_text_processor,
        test_without_model,
        test_api_models,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== 测试结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("✅ 所有基础功能正常！")
        print("\n要完整测试服务，请:")
        print("1. 运行 'python main.py' 启动服务")
        print("2. 访问 http://localhost:8000/docs 查看API文档")
        print("3. 运行 'python examples/api_client.py' 测试API")
    else:
        print("❌ 部分功能存在问题，请检查代码")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())