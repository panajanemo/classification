import pytest
from src.utils.text_processor import TextProcessor


class TestTextProcessor:
    """文本处理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = TextProcessor()
    
    def test_split_sentences_chinese(self):
        """测试中文分句"""
        text = "这是第一句话。这是第二句话！这是第三句话？"
        sentences = self.processor.split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "这是第一句话。"
        assert sentences[1] == "这是第二句话！"
        assert sentences[2] == "这是第三句话？"
    
    def test_split_sentences_mixed(self):
        """测试中英文混合分句"""
        text = "这是中文句子。This is English. 这是另一个中文句子！"
        sentences = self.processor.split_sentences(text)
        
        assert len(sentences) >= 2
        assert "这是中文句子。" in sentences[0]
        assert "这是另一个中文句子！" in sentences[-1]
    
    def test_split_sentences_empty(self):
        """测试空文本分句"""
        assert self.processor.split_sentences("") == []
        assert self.processor.split_sentences("   ") == []
        assert self.processor.split_sentences(None) == []
    
    def test_clean_text(self):
        """测试文本清洗"""
        dirty_text = "这是一段   有很多空格的    文本！！！"
        clean_text = self.processor.clean_text(dirty_text)
        
        assert "   " not in clean_text
        assert clean_text.strip() == clean_text
    
    def test_clean_text_special_chars(self):
        """测试特殊字符清洗"""
        text_with_special = "这是文本@#$%含有特殊字符★☆的内容。"
        clean_text = self.processor.clean_text(text_with_special)
        
        assert "@#$%" not in clean_text
        assert "★☆" not in clean_text
        assert "这是文本" in clean_text
        assert "含有特殊字符" in clean_text
    
    def test_normalize_text(self):
        """测试文本标准化"""
        text = "这是一段  需要标准化的文本 。 它有不规范的标点！这里没有空格。"
        normalized = self.processor.normalize_text(text)
        
        assert "  " not in normalized
        assert "。 " not in normalized or normalized.count("。 ") <= 1
    
    def test_validate_input_valid(self):
        """测试有效输入验证"""
        valid_text = "这是一段有效的中文文本。"
        assert self.processor.validate_input(valid_text) == True
    
    def test_validate_input_invalid(self):
        """测试无效输入验证"""
        assert self.processor.validate_input("") == False
        assert self.processor.validate_input("   ") == False
        assert self.processor.validate_input(None) == False
        assert self.processor.validate_input(123) == False
    
    def test_validate_input_too_long(self):
        """测试过长文本验证"""
        long_text = "很长的文本" * 2000
        assert self.processor.validate_input(long_text, max_length=100) == False
        assert self.processor.validate_input(long_text, max_length=20000) == True
    
    def test_format_output(self):
        """测试输出格式化"""
        paragraphs = ["第一段", "第二段", "第三段"]
        output = self.processor.format_output(paragraphs)
        
        assert "第一段" in output
        assert "第二段" in output
        assert "第三段" in output
        assert output.count("\n\n") == 2
    
    def test_format_output_custom_separator(self):
        """测试自定义分隔符输出"""
        paragraphs = ["段落1", "段落2"]
        output = self.processor.format_output(paragraphs, separator=" | ")
        
        assert output == "段落1 | 段落2"
    
    def test_format_output_empty(self):
        """测试空段落输出"""
        assert self.processor.format_output([]) == ""
        assert self.processor.format_output(["", "  ", ""]) == ""
        assert self.processor.format_output(["有内容", "", "另一段"]) == "有内容\n\n另一段"