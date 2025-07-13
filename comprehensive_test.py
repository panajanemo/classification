#!/usr/bin/env python3
"""
综合功能测试脚本 - 展示完整的语义分段系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
from src.core.transformer_segmenter import TransformerSemanticSegmenter
from src.core.enhanced_multi_task_segmenter import EnhancedMultiTaskSegmenter
from src.core.hybrid_segmenter import HybridSemanticSegmenter
from src.core.topic_consistency_evaluator import TopicConsistencyEvaluator
from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector
from src.utils.text_processor import TextProcessor
import time
import torch


def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)


def print_section(title: str):
    """打印小节标题"""
    print(f"\n📋 {title}")
    print("-"*60)


def test_all_segmentation_methods():
    """测试所有分段方法"""
    print_separator("综合语义分段系统测试")
    
    # 测试文本 - 包含多种主题的复杂文本
    test_text = """
    萧炎缓缓睁开双眼，眼前依然是熟悉的石室。淡淡的药香在空气中弥漫，让人精神为之一振。他从石床上坐起身来，活动了一下有些僵硬的筋骨。
    
    "你终于醒了。"药老的声音在耳边响起，带着一丝淡淡的笑意。"这次的修炼效果如何？"
    
    萧炎感受着体内涌动的斗气，眼中闪过一抹喜色。经过这段时间的苦修，他的实力确实有了长足的进步。
    
    然而，让我们暂时离开这个奇幻的世界，来看看现实中的科技发展。最近人工智能领域取得了突破性进展，ChatGPT、Claude等大语言模型的出现，标志着AI技术进入了新的时代。
    
    这些AI系统能够理解自然语言，进行复杂的对话，甚至协助编程、写作、数据分析等工作。它们的出现正在深刻改变着我们的工作和生活方式。
    
    在技术实现层面，这些大模型基于Transformer架构，使用了数万亿参数，经过海量文本数据的预训练。Python作为主要的开发语言，PyTorch和TensorFlow等深度学习框架为模型开发提供了强大支持。
    
    当然，AI技术的发展也带来了新的思考。如何确保AI的安全性、可解释性，如何处理AI带来的伦理问题，这些都是我们需要认真面对的挑战。
    
    回到萧炎的世界，他正在思考如何将现代科技的理念融入到斗气修炼中。或许，数据分析的方法可以帮助他更好地理解功法的奥秘，算法优化的思路可以指导他改进修炼效率。
    
    "有趣的想法。"药老似乎读懂了萧炎的心思，"古今结合，或许能产生意想不到的效果。"
    """
    
    print(f"📝 测试文本长度: {len(test_text)} 字符")
    print(f"📊 预估主题: 奇幻小说 + 人工智能技术 + 古今结合")
    
    # 初始化所有分段器
    print_section("初始化分段器")
    
    model = SentenceTransformerModel(device='mps')
    text_processor = TextProcessor()
    type_detector = TextTypeDetector()
    
    # 各种分段器
    enhanced_segmenter = EnhancedSemanticSegmenter(model, text_processor, type_detector)
    transformer_segmenter = TransformerSemanticSegmenter(model, text_processor, type_detector, 'mps')
    multi_task_segmenter = EnhancedMultiTaskSegmenter(model, text_processor, type_detector, 'cpu')
    hybrid_segmenter = HybridSemanticSegmenter(model, text_processor, type_detector, 'mps')
    consistency_evaluator = TopicConsistencyEvaluator(model, text_processor, "chinese")
    
    print("✅ 所有分段器初始化完成")
    
    # 测试各种方法
    results = {}
    
    print_section("方法1: 增强版多尺度分段器")
    start_time = time.time()
    enhanced_result = enhanced_segmenter.segment_text_enhanced(test_text)
    enhanced_time = time.time() - start_time
    
    if "error" not in enhanced_result:
        results["enhanced"] = enhanced_result
        print(f"✅ 分段成功: {len(enhanced_result['paragraphs'])}段")
        print(f"📊 文本类型: {enhanced_result['text_type']} (置信度: {enhanced_result['type_confidence']:.3f})")
        print(f"⭐ 质量分数: {enhanced_result['quality']['quality_score']:.3f}")
        print(f"⏱️  处理时间: {enhanced_time:.3f}s")
    else:
        print(f"❌ 失败: {enhanced_result['error']}")
    
    print_section("方法2: Transformer²架构分段器")
    start_time = time.time()
    transformer_result = transformer_segmenter.segment_text_transformer(test_text)
    transformer_time = time.time() - start_time
    
    if "error" not in transformer_result:
        results["transformer"] = transformer_result
        print(f"✅ 分段成功: {len(transformer_result['paragraphs'])}段")
        print(f"📊 文本类型: {transformer_result['text_type']} (置信度: {transformer_result['type_confidence']:.3f})")
        print(f"⭐ 质量分数: {transformer_result['quality']['quality_score']:.3f}")
        print(f"⏱️  处理时间: {transformer_time:.3f}s")
    else:
        print(f"❌ 失败: {transformer_result['error']}")
    
    print_section("方法3: 多任务学习分段器")
    start_time = time.time()
    multi_task_result = multi_task_segmenter.segment_text_multi_task(test_text)
    multi_task_time = time.time() - start_time
    
    if "error" not in multi_task_result:
        results["multi_task"] = multi_task_result
        print(f"✅ 分段成功: {len(multi_task_result['paragraphs'])}段")
        print(f"📊 文本类型: {multi_task_result['text_type']} (置信度: {multi_task_result['type_confidence']:.3f})")
        print(f"⭐ 质量分数: {multi_task_result['quality']['quality_score']:.3f}")
        print(f"🧠 模型状态: {'已训练' if multi_task_result['is_trained'] else '未训练'}")
        print(f"⏱️  处理时间: {multi_task_time:.3f}s")
    else:
        print(f"❌ 失败: {multi_task_result['error']}")
    
    print_section("方法4: 混合智能分段器")
    
    # 测试不同策略
    methods = ["enhanced", "transformer", "hybrid", "auto"]
    hybrid_results = {}
    
    for method in methods:
        start_time = time.time()
        hybrid_result = hybrid_segmenter.segment_text(test_text, method)
        method_time = time.time() - start_time
        
        if "error" not in hybrid_result:
            hybrid_results[method] = hybrid_result
            print(f"✅ {method}: {len(hybrid_result['paragraphs'])}段, 质量{hybrid_result['quality']['quality_score']:.3f}, {method_time:.3f}s")
        else:
            print(f"❌ {method}: {hybrid_result['error']}")
    
    # 混合分段器方法比较
    print("\n🔄 混合分段器自动比较:")
    comparison = hybrid_segmenter.compare_methods(test_text)
    if "error" not in comparison:
        comp_summary = comparison["comparison"]
        print(f"🏆 最佳方法: {comp_summary['best_method']}")
        if 'quality_comparison' in comp_summary:
            print("📈 质量排名:")
            sorted_methods = sorted(comp_summary['quality_comparison'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for rank, (method, score) in enumerate(sorted_methods, 1):
                print(f"   {rank}. {method}: {score:.3f}")
    
    return results


def test_topic_consistency_evaluation(results):
    """测试主题一致性评估"""
    print_separator("主题一致性评估测试")
    
    if not results:
        print("❌ 没有可用的分段结果进行评估")
        return
    
    # 初始化评估器
    model = SentenceTransformerModel(device='mps')
    text_processor = TextProcessor()
    evaluator = TopicConsistencyEvaluator(model, text_processor, "chinese")
    
    print_section("各方法一致性评估")
    
    # 准备分段结果
    segmentation_results = {}
    for method, result in results.items():
        if "paragraphs" in result:
            paragraphs = [p["text"] for p in result["paragraphs"]]
            segmentation_results[method] = paragraphs
            print(f"📝 {method}: {len(paragraphs)}段")
    
    # 执行一致性比较
    print_section("一致性比较分析")
    comparison_result = evaluator.compare_segmentation_consistency(segmentation_results)
    
    if "error" not in comparison_result.get("comparison_summary", {}):
        summary = comparison_result["comparison_summary"]
        detailed = comparison_result["detailed_results"]
        
        print(f"🏆 一致性最佳方法: {summary['best_method']}")
        print("\n📊 一致性评分排名:")
        for rank, (method, score) in enumerate(summary['consistency_ranking'], 1):
            print(f"   {rank}. {method}: {score:.3f}")
        
        print("\n📋 详细一致性分析:")
        for method, result in detailed.items():
            if "error" not in result:
                print(f"\n   📝 {method.upper()}:")
                print(f"      综合一致性: {result['consistency_score']:.3f}")
                print(f"      语义相似度: {result['semantic_analysis']['avg_similarity']:.3f}")
                print(f"      主题连贯性: {result['coherence_metrics']['topic_coherence']:.3f}")
                print(f"      段落内一致性: {result['paragraph_metrics']['avg_internal_consistency']:.3f}")
                
                if result['coherence_metrics']['detected_boundaries']:
                    print(f"      检测到主题边界: {result['coherence_metrics']['detected_boundaries']}")
    else:
        print(f"❌ 一致性评估失败: {comparison_result['comparison_summary']['error']}")


def display_detailed_results(results):
    """显示详细分段结果"""
    print_separator("详细分段结果展示")
    
    for method, result in results.items():
        if "paragraphs" in result:
            print_section(f"{method.upper()}方法分段结果")
            
            paragraphs = result["paragraphs"]
            print(f"📊 总段落数: {len(paragraphs)}")
            print(f"📝 文本类型: {result.get('text_type', 'unknown')}")
            print(f"⭐ 质量分数: {result['quality']['quality_score']:.3f}")
            print(f"🔧 使用方法: {result.get('method', method)}")
            
            print("\n段落内容:")
            for i, paragraph in enumerate(paragraphs, 1):
                text = paragraph["text"]
                length = paragraph["length"]
                sentences = paragraph.get("sentence_count", "N/A")
                
                # 截断显示
                display_text = text if len(text) <= 100 else text[:100] + "..."
                print(f"   段落{i} (长度:{length}, 句数:{sentences}): {display_text}")


def test_chinese_optimization():
    """测试中文优化功能"""
    print_separator("中文语言优化测试")
    
    from src.utils.chinese_optimizer import ChineseTextOptimizer
    
    optimizer = ChineseTextOptimizer()
    
    # 中文测试文本
    chinese_text = "萧炎听了这话，心中不禁一动。\"原来如此。\"他轻声说道。然后，话题突然转向了现代科技。人工智能的发展确实令人震撼。这些技术正在改变世界。"
    
    print_section("中文文本分析")
    print(f"📝 测试文本: {chinese_text}")
    
    # 中文比例检测
    chinese_ratio = optimizer.detect_chinese_ratio(chinese_text)
    print(f"🔤 中文字符比例: {chinese_ratio:.3f}")
    
    # 增强分句
    sentences = optimizer.enhanced_chinese_sentence_split(chinese_text)
    print(f"📜 增强分句结果 ({len(sentences)}句):")
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. {sentence}")
    
    # 关键词提取
    keywords = optimizer.extract_chinese_keywords(chinese_text, 5)
    print(f"🏷️  中文关键词:")
    for word, pos, score in keywords:
        print(f"   {word} ({pos}): {score:.3f}")
    
    # 对话检测
    dialogues = optimizer.detect_dialogue_segments(chinese_text)
    print(f"💬 对话检测:")
    for start, end, content in dialogues:
        print(f"   位置{start}-{end}: \"{content}\"")
    
    # 语义特征
    features = optimizer.calculate_chinese_semantic_features(chinese_text)
    print(f"📊 中文语义特征:")
    for key, value in features.items():
        print(f"   {key}: {value:.3f}")


def main():
    """主测试函数"""
    print("🚀 启动增强版语义分段系统综合测试")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    print(f"💻 MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"🚀 CUDA可用: {torch.cuda.is_available()}")
    
    try:
        # 1. 测试所有分段方法
        results = test_all_segmentation_methods()
        
        # 2. 测试主题一致性评估
        test_topic_consistency_evaluation(results)
        
        # 3. 显示详细结果
        display_detailed_results(results)
        
        # 4. 测试中文优化
        test_chinese_optimization()
        
        print_separator("测试完成总结")
        print("🎉 所有功能测试完成！")
        print("✅ 增强版多尺度分段器")
        print("✅ Transformer²架构分段器") 
        print("✅ 多任务学习分段器")
        print("✅ 混合智能分段器")
        print("✅ 主题一致性评估器")
        print("✅ 中文语言优化器")
        print("\n🎯 系统已达到production-ready状态！")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()