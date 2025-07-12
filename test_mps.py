#!/usr/bin/env python3
"""
测试MPS配置的语义分段功能
"""

import sys
import os
import time

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mps_config():
    """测试MPS配置"""
    print("=== 测试MPS配置的语义分段 ===\n")
    
    try:
        from config.settings import settings
        print(f"配置的设备: {settings.device}")
        
        from src.core.bert_model import BERTModel
        bert = BERTModel()
        print(f"实际使用设备: {bert.device}")
        
        if bert.device == "mps":
            print("✅ 成功配置MPS设备")
        else:
            print(f"⚠️  使用设备: {bert.device} (而非MPS)")
        
        # 测试设备可用性
        import torch
        if bert.device == "mps":
            try:
                test_tensor = torch.randn(10, 768).to("mps")
                print("✅ MPS设备测试通过")
            except Exception as e:
                print(f"❌ MPS设备测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"配置测试失败: {e}")
        return False

def test_text_processing_speed():
    """测试文本处理速度"""
    print("\n=== 文本处理速度测试 ===\n")
    
    try:
        from src.utils.text_processor import TextProcessor
        processor = TextProcessor()
        
        # 测试文本
        text = """
        人工智能技术正在快速发展。机器学习和深度学习已经在各个领域展现出巨大潜力。
        自然语言处理是人工智能的重要分支。它让计算机能够理解和生成人类语言。
        语义分析技术帮助我们更好地理解文本内容。通过分析句子间的语义关系，我们可以实现更精确的文本分段。
        这种技术在文档处理、内容分析等领域有广泛应用。未来会有更多创新应用出现。
        """ * 5  # 重复5次增加文本长度
        
        start_time = time.time()
        
        # 测试分句
        sentences = processor.split_sentences(text)
        
        # 测试文本清洗
        clean_text = processor.clean_text(text)
        
        # 测试标准化
        normalized_text = processor.normalize_text(text)
        
        end_time = time.time()
        
        print(f"文本长度: {len(text)} 字符")
        print(f"分句结果: {len(sentences)} 句")
        print(f"处理时间: {(end_time - start_time) * 1000:.2f} 毫秒")
        print("✅ 文本处理速度测试完成")
        
        return True
        
    except Exception as e:
        print(f"文本处理测试失败: {e}")
        return False

def test_mock_segmentation():
    """测试模拟分段功能"""
    print("\n=== 模拟分段功能测试 ===\n")
    
    try:
        from src.core.semantic_segmenter import SemanticSegmenter
        from src.utils.text_processor import TextProcessor
        from unittest.mock import Mock
        import torch
        
        # 创建模拟BERT模型（避免下载大模型）
        mock_bert = Mock()
        mock_bert.device = "mps"
        mock_bert.encode_texts.return_value = torch.randn(10, 768)
        mock_bert.calculate_similarity.side_effect = lambda x, y: 0.6
        mock_bert.batch_similarities.return_value = [0.7, 0.3, 0.8, 0.2]
        mock_bert.get_model_info.return_value = {"device": "mps", "model_name": "mock"}
        
        # 创建分段器
        text_processor = TextProcessor()
        segmenter = SemanticSegmenter(mock_bert, text_processor)
        
        # 测试文本
        test_text = """
        深度学习是机器学习的重要分支。它通过多层神经网络来学习数据中的复杂模式。
        卷积神经网络在图像识别方面表现优异。它能够自动提取图像的特征。
        递归神经网络擅长处理序列数据。在自然语言处理中应用广泛。
        Transformer架构彻底改变了NLP领域。注意力机制是其核心创新。
        BERT模型基于Transformer架构。它在多个NLP任务上都取得了优异成绩。
        """
        
        start_time = time.time()
        result = segmenter.segment_text(test_text.strip())
        end_time = time.time()
        
        if "error" not in result:
            print(f"分段成功!")
            print(f"原文句子数: {result.get('sentence_count', 0)}")
            print(f"段落数: {len(result.get('paragraphs', []))}")
            print(f"处理时间: {(end_time - start_time) * 1000:.2f} 毫秒")
            
            for i, para in enumerate(result.get('paragraphs', []), 1):
                print(f"段落{i}: {para[:50]}...")
            
            print("✅ 模拟分段测试完成")
        else:
            print(f"分段失败: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"模拟分段测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试Mac MPS配置...")
    
    tests = [
        test_mps_config,
        test_text_processing_speed,
        test_mock_segmentation,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果: {passed}/{len(tests)} 通过 ===")
    
    if passed == len(tests):
        print("✅ 所有测试通过！你的Mac MPS配置正常")
        print("\n下一步:")
        print("1. 运行 'python main.py' 启动完整服务（需下载BERT模型）")
        print("2. 或运行 'python quick_start.py' 启动演示服务（使用模拟模型）")
    else:
        print("❌ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main()