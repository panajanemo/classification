#!/usr/bin/env python3
"""
基本使用示例
"""

import sys
import os
import json

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.semantic_segmenter import SemanticSegmenter
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor


def main():
    """主函数"""
    print("=== BERT语义分段服务 - 基本使用示例 ===\n")
    
    # 示例文本
    text = """
    人工智能是计算机科学的一个重要分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    机器学习是人工智能的核心，是使计算机具有智能的根本途径。
    深度学习又是机器学习的一个重要分支，它模拟人脑的神经网络结构。
    自从人工智能诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    然而，人工智能的发展也带来了一些挑战和担忧。
    比如就业问题、隐私安全、算法偏见等都需要我们认真对待。
    """
    
    try:
        print("正在初始化语义分段器...")
        
        # 创建组件
        bert_model = BERTModel()
        text_processor = TextProcessor()
        segmenter = SemanticSegmenter(bert_model, text_processor)
        
        print("正在加载BERT模型...")
        # 预加载模型
        bert_model.load_model()
        
        print("开始文本分段...\n")
        
        # 执行分段
        result = segmenter.segment_text(text.strip())
        
        if "error" in result:
            print(f"分段失败: {result['error']}")
            return
        
        # 显示结果
        print("=== 分段结果 ===")
        for i, paragraph in enumerate(result["paragraphs"], 1):
            print(f"段落 {i}:")
            print(f"  {paragraph}")
            print()
        
        print("=== 统计信息 ===")
        print(f"原文句子数: {result.get('sentence_count', 0)}")
        print(f"段落数: {len(result['paragraphs'])}")
        print(f"段落边界数: {result.get('boundary_count', 0)}")
        
        # 显示质量指标
        if "quality" in result:
            quality = result["quality"]
            print("\n=== 质量指标 ===")
            print(f"平均段落长度: {quality.get('avg_length', 0):.1f} 字符")
            print(f"长度标准差: {quality.get('length_std', 0):.1f}")
            print(f"语义一致性: {quality.get('semantic_consistency', 0):.3f}")
            print(f"质量分数: {quality.get('quality_score', 0):.3f}")
        
        # 格式化输出
        formatted_text = text_processor.format_output(result["paragraphs"])
        print("\n=== 格式化输出 ===")
        print(formatted_text)
        
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()