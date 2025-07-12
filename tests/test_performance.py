import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import sys
import os

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.semantic_segmenter import SemanticSegmenter
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor


class TestPerformance:
    """性能测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 使用模拟组件避免实际模型加载
        self.mock_bert = Mock(spec=BERTModel)
        self.text_processor = TextProcessor()
        self.segmenter = SemanticSegmenter(self.mock_bert, self.text_processor)
        
        # 模拟BERT编码速度
        import torch
        self.mock_bert.encode_texts.return_value = torch.randn(10, 768)
        self.mock_bert.calculate_similarity.return_value = 0.5
        self.mock_bert.batch_similarities.return_value = [0.5] * 9
    
    def generate_test_text(self, sentence_count: int = 10) -> str:
        """生成测试文本"""
        sentences = []
        for i in range(sentence_count):
            if i % 3 == 0:
                sentences.append(f"这是关于主题A的第{i+1}句话。")
            elif i % 3 == 1:
                sentences.append(f"这是关于主题B的第{i+1}句话。")
            else:
                sentences.append(f"这是关于主题C的第{i+1}句话。")
        return "".join(sentences)
    
    def test_single_request_performance(self):
        """测试单次请求性能"""
        text = self.generate_test_text(20)
        
        start_time = time.time()
        result = self.segmenter.segment_text(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 验证响应时间（模拟环境下应该很快）
        assert processing_time < 1.0, f"处理时间过长: {processing_time}s"
        assert "paragraphs" in result
        print(f"单次请求处理时间: {processing_time:.3f}s")
    
    def test_multiple_requests_sequential(self):
        """测试顺序多请求性能"""
        texts = [self.generate_test_text(15) for _ in range(5)]
        
        start_time = time.time()
        results = []
        for text in texts:
            result = self.segmenter.segment_text(text)
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(texts)
        
        assert len(results) == 5
        assert all("paragraphs" in r for r in results)
        print(f"5次顺序请求总时间: {total_time:.3f}s, 平均: {avg_time:.3f}s")
    
    def test_concurrent_requests(self):
        """测试并发请求性能"""
        texts = [self.generate_test_text(10) for _ in range(10)]
        
        def process_text(text):
            """处理单个文本"""
            start = time.time()
            result = self.segmenter.segment_text(text)
            end = time.time()
            return result, end - start
        
        start_time = time.time()
        
        # 使用线程池进行并发测试
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_text, text) for text in texts]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有请求都成功
        assert len(results) == 10
        successful_results = [r for r, t in results if "paragraphs" in r]
        assert len(successful_results) == 10
        
        # 计算性能指标
        processing_times = [t for r, t in results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        print(f"并发测试结果:")
        print(f"  总时间: {total_time:.3f}s")
        print(f"  平均处理时间: {avg_processing_time:.3f}s")
        print(f"  最大处理时间: {max_processing_time:.3f}s")
        
        # 并发应该比顺序处理更快（在真实环境中）
        # 在模拟环境中主要验证没有错误
        assert avg_processing_time < 1.0
    
    def test_text_length_scalability(self):
        """测试文本长度可扩展性"""
        sentence_counts = [5, 10, 20, 50]
        processing_times = []
        
        for count in sentence_counts:
            text = self.generate_test_text(count)
            
            start_time = time.time()
            result = self.segmenter.segment_text(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            assert "paragraphs" in result
            print(f"句子数: {count}, 处理时间: {processing_time:.3f}s")
        
        # 验证处理时间随文本长度合理增长
        # 在模拟环境中，时间增长可能不明显
        assert all(t < 2.0 for t in processing_times)
    
    def test_memory_usage_simulation(self):
        """测试内存使用模拟"""
        # 模拟大量文本处理
        large_texts = [self.generate_test_text(30) for _ in range(20)]
        
        results = []
        for text in large_texts:
            result = self.segmenter.segment_text(text)
            results.append(result)
            
            # 验证结果质量
            if "paragraphs" in result:
                assert len(result["paragraphs"]) > 0
        
        # 验证所有处理都成功
        successful_count = sum(1 for r in results if "paragraphs" in r)
        assert successful_count == len(large_texts)
        print(f"成功处理了 {successful_count} 个大文本")
    
    def test_error_handling_performance(self):
        """测试错误处理性能"""
        invalid_inputs = ["", "   ", None, "a" * 20000]
        
        start_time = time.time()
        for invalid_input in invalid_inputs:
            try:
                if invalid_input is None:
                    continue  # 跳过None输入测试
                result = self.segmenter.segment_text(invalid_input)
                # 错误应该被优雅处理
                assert "error" in result or "paragraphs" in result
            except Exception as e:
                # 异常应该被捕获和处理
                assert False, f"未处理的异常: {e}"
        
        end_time = time.time()
        error_handling_time = end_time - start_time
        
        # 错误处理应该很快
        assert error_handling_time < 0.1
        print(f"错误处理时间: {error_handling_time:.3f}s")
    
    def test_threshold_adjustment_performance(self):
        """测试阈值调整性能"""
        text = self.generate_test_text(15)
        thresholds = [0.2, 0.4, 0.6, 0.8]
        
        results = []
        start_time = time.time()
        
        for threshold in thresholds:
            self.segmenter.set_threshold(threshold)
            result = self.segmenter.segment_text(text)
            results.append((threshold, result))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有阈值都产生了结果
        assert len(results) == 4
        for threshold, result in results:
            assert "paragraphs" in result
        
        print(f"阈值调整测试总时间: {total_time:.3f}s")
        
        # 阈值调整应该很快
        assert total_time < 2.0
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_performance(self, mock_cuda):
        """测试CPU性能"""
        # 确保使用CPU
        assert not torch.cuda.is_available()
        
        text = self.generate_test_text(20)
        start_time = time.time()
        result = self.segmenter.segment_text(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert "paragraphs" in result
        print(f"CPU处理时间: {processing_time:.3f}s")
    
    def test_batch_processing_efficiency(self):
        """测试批处理效率"""
        # 模拟批处理多个文本
        texts = [self.generate_test_text(8) for _ in range(5)]
        
        # 顺序处理
        start_time = time.time()
        sequential_results = []
        for text in texts:
            result = self.segmenter.segment_text(text)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # 在实际场景中，可以实现真正的批处理
        # 这里我们只是验证功能正确性
        assert len(sequential_results) == 5
        assert all("paragraphs" in r for r in sequential_results)
        
        print(f"批处理模拟时间: {sequential_time:.3f}s")