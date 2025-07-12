#!/usr/bin/env python3
"""
测试基础组件而不加载实际模型
"""

from src.core.text_type_detector import TextTypeDetector, TextType
from src.utils.text_processor import TextProcessor


def test_text_type_detector():
    """测试文本类型检测器"""
    print("🔍 测试文本类型检测器...")
    
    detector = TextTypeDetector()
    
    test_cases = [
        {
            "name": "技术文档",
            "text": "Python是一种编程语言。它具有函数、类、API接口等概念。我们可以使用import语句来导入模块。def定义函数，class定义类。"
        },
        {
            "name": "小说文本", 
            "text": "萧薰儿美丽动人，她的眼中闪烁着温柔的光芒。少年萧炎望着她，心中充满了复杂的情感。他说道：\"薰儿，你还记得我们的约定吗？\""
        },
        {
            "name": "学术论文",
            "text": "本研究采用实验方法分析了不同变量对结果的影响。根据数据统计，我们发现显著性差异。因此，假设得到验证。参考文献显示类似的研究结果。"
        }
    ]
    
    for case in test_cases:
        text_type, scores = detector.detect_text_type(case["text"])
        config = detector.get_segmentation_config(text_type)
        
        print(f"\n📄 {case['name']}:")
        print(f"   检测类型: {text_type.value}")
        print(f"   置信度: {scores['confidence']:.3f}")
        print(f"   推荐阈值: {config['threshold']}")
        print(f"   窗口大小: {config['window_size']}")
        print(f"   最小段落长度: {config['min_paragraph_length']}")


def test_text_processor():
    """测试文本处理器"""
    print("\n✂️ 测试文本处理器...")
    
    processor = TextProcessor()
    
    test_text = """
    这是第一句话。这是第二句话！这是第三句话？
    这是第四句话，包含逗号。这是第五句话；包含分号。
    
    这是一个新段落的开始。
    """
    
    # 测试文本规范化
    normalized = processor.normalize_text(test_text)
    print(f"📝 规范化文本: {repr(normalized)}")
    
    # 测试分句
    sentences = processor.split_sentences(normalized)
    print(f"🔢 分句结果 ({len(sentences)} 句):")
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. {sentence}")
    
    # 测试格式化输出
    paragraphs = ["第一段内容", "第二段内容", "第三段内容"]
    formatted = processor.format_output(paragraphs)
    print(f"📄 格式化输出:\n{formatted}")


if __name__ == "__main__":
    test_text_type_detector()
    test_text_processor()
    print("\n✅ 基础组件测试完成！")