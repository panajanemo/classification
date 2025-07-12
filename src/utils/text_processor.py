import re
import jieba
from typing import List, Optional


class TextProcessor:
    """文本预处理器"""
    
    def __init__(self):
        # 中文句号、问号、感叹号等句子结束符
        self.sentence_endings = r'[。！？；：]'
        # 英文句子结束符
        self.english_endings = r'[.!?;:]'
        
    def split_sentences(self, text: str) -> List[str]:
        """
        中文分句
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if not text or not text.strip():
            return []
            
        # 预处理：规范化空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 使用正则表达式分句
        sentences = []
        
        # 先按中文标点分句
        parts = re.split(f'({self.sentence_endings})', text)
        
        current_sentence = ""
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            current_sentence += part
            
            # 如果是句子结束符，完成当前句子
            if re.match(self.sentence_endings, part):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            # 如果是最后一部分且不以标点结尾，也作为句子
            elif i == len(parts) - 1 and current_sentence.strip():
                sentences.append(current_sentence.strip())
        
        # 过滤空句子和过短句子
        sentences = [s for s in sentences if len(s.strip()) > 3]
        
        return sentences
    
    def clean_text(self, text: str) -> str:
        """
        文本清洗
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
            
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 简化清洗，只做基本处理
        # 规范化引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        文本标准化
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
            
        # 清洗文本
        text = self.clean_text(text)
        
        # 规范化标点符号间距
        text = re.sub(r'\s*([。！？；：，、])\s*', r'\1', text)
        
        # 确保句子间有适当间距
        text = re.sub(r'([。！？])([^\s])', r'\1 \2', text)
        
        return text
    
    def validate_input(self, text: str, max_length: int = 10000) -> bool:
        """
        验证输入文本
        
        Args:
            text: 输入文本
            max_length: 最大长度限制
            
        Returns:
            是否有效
        """
        if not text or not isinstance(text, str):
            return False
            
        if len(text.strip()) == 0:
            return False
            
        if len(text) > max_length:
            return False
            
        return True
    
    def format_output(self, paragraphs: List[str], separator: str = "\n\n") -> str:
        """
        格式化输出
        
        Args:
            paragraphs: 段落列表
            separator: 段落分隔符
            
        Returns:
            格式化后的文本
        """
        if not paragraphs:
            return ""
            
        # 过滤空段落
        valid_paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
        
        return separator.join(valid_paragraphs)