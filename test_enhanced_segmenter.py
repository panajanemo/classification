#!/usr/bin/env python3
"""
测试增强版语义分段器
"""

import time
from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector
from src.utils.text_processor import TextProcessor


def test_enhanced_segmenter():
    """测试增强版分段器"""
    
    print("🚀 初始化增强版语义分段器...")
    
    # 初始化组件
    try:
        sentence_model = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2")
        text_processor = TextProcessor()
        type_detector = TextTypeDetector()
        enhanced_segmenter = EnhancedSemanticSegmenter(sentence_model, text_processor, type_detector)
        
        print("✅ 组件初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试文本
    test_texts = [
        {
            "name": "技术文档",
            "text": """
            Python是一种强大的编程语言，具有简洁的语法和丰富的库生态系统。它广泛应用于Web开发、数据科学和人工智能领域。
            
            机器学习是人工智能的核心分支，通过算法让计算机从数据中学习模式。深度学习使用神经网络来处理复杂的数据结构。
            
            BERT模型是基于Transformer架构的预训练语言模型。它在自然语言处理任务中表现出色，特别是在文本分类和问答系统中。
            
            API设计是软件开发的重要环节。RESTful API提供了标准化的接口设计方式，使得不同系统之间能够有效地进行数据交互。
            """
        },
        {
            "name": "小说文本",
            "text": """
            夕阳西下，萧薰儿站在山顶上，目光眺望着远方的云海。微风轻抚过她的长发，带来阵阵花香。
            
            "萧炎哥哥，你还记得小时候我们在这里许下的诺言吗？"她轻声说道，声音中带着一丝颤抖。
            
            远处的天空中，一群大雁正向南飞去。它们的鸣叫声在山谷中回荡，显得格外悠远。薰儿的眼中闪烁着泪光，那是对过往岁月的眷恋。
            
            时光荏苒，当年的少年如今已经成长为顶天立地的男子汉。但那份初心，那份真挚的感情，依然如当初一般纯净。
            """
        }
    ]
    
    print(f"\n📝 开始测试 {len(test_texts)} 个文本样本...")
    
    for i, sample in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {sample['name']}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            # 执行增强版分段
            result = enhanced_segmenter.segment_text_enhanced(sample['text'])
            
            processing_time = time.time() - start_time
            
            if "error" in result:
                print(f"❌ 分段失败: {result['error']}")
                continue
            
            # 显示结果
            print(f"📊 分段结果:")
            print(f"   检测文本类型: {result['text_type']} (置信度: {result['type_confidence']:.3f})")
            print(f"   原始句子数: {result['sentence_count']}")
            print(f"   分段边界数: {result['boundary_count']}")
            print(f"   生成段落数: {len(result['paragraphs'])}")
            print(f"   处理时间: {processing_time:.3f}秒")
            
            print(f"\n📚 分段内容:")
            for j, para in enumerate(result['paragraphs'], 1):
                print(f"\n段落 {j} [{para['type']}]:")
                print(f"   内容: {para['text'][:100]}{'...' if len(para['text']) > 100 else ''}")
                print(f"   长度: {para['length']} 字符, {para['sentence_count']} 句")
                if para['key_phrases']:
                    print(f"   关键词: {', '.join(para['key_phrases'])}")
            
            print(f"\n📈 质量评估:")
            quality = result['quality']
            print(f"   综合质量分数: {quality['quality_score']:.3f}")
            print(f"   语义一致性: {quality['semantic_consistency']:.3f}")
            print(f"   结构合理性: {quality['structure_score']:.3f}")
            print(f"   段落类型分布: {quality['type_distribution']}")
            
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_segmenter()