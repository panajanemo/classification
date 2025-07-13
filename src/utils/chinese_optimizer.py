import logging
import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import numpy as np

from src.utils.text_processor import TextProcessor


logger = logging.getLogger(__name__)


class ChineseTextOptimizer:
    """
    中文文本处理优化器
    专门针对中文文本的语义分段优化
    """
    
    def __init__(self):
        """初始化中文文本优化器"""
        
        # 中文标点符号
        self.chinese_punctuation = {
            '。', '！', '？', '；', '：', '，', '、', '"', '"', ''', ''',
            '（', '）', '【', '】', '《', '》', '〈', '〉', '「', '」',
            '『', '』', '〔', '〕', '［', '］', '｛', '｝', '…', '——',
            '·', '～', '￥', '％'
        }
        
        # 中文句子结束标记
        self.sentence_endings = {'。', '！', '？', '；'}
        
        # 中文段落转换词汇
        self.transition_words = {
            # 时间转换
            '后来', '接着', '然后', '随后', '接下来', '此时', '这时', '那时',
            '突然', '忽然', '瞬间', '刹那', '顷刻', '片刻', '不久',
            
            # 逻辑转换
            '因此', '所以', '但是', '可是', '然而', '不过', '虽然', '尽管',
            '另外', '此外', '而且', '并且', '同时', '与此同时',
            
            # 场景转换
            '与此同时', '在另一边', '另一方面', '回到', '转眼', '话说',
            '说起', '提到', '关于', '至于'
        }
        
        # 中文语篇标记
        self.discourse_markers = {
            '首先', '其次', '再次', '最后', '总之', '综上所述',
            '第一', '第二', '第三', '最后一点',
            '一方面', '另一方面', '总的来说', '换句话说'
        }
        
        # 中文对话标识
        self.dialogue_patterns = [
            r'[""]([^""]*?)[""]',  # 双引号对话
            r"['']([^'']*?)['']",  # 单引号对话
            r'「([^」]*?)」',      # 日式引号
            r'『([^』]*?)』',      # 中式书名号对话
            r'([^："]*?)说[：:][""]',  # XX说："...
            r'([^："]*?)道[：:][""]',  # XX道："...
        ]
        
        # 初始化jieba
        jieba.initialize()
        
        logger.info("中文文本优化器初始化完成")
    
    def detect_chinese_ratio(self, text: str) -> float:
        """检测文本中中文字符的比例"""
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if not char.isspace():
                total_chars += 1
                if '\u4e00' <= char <= '\u9fff':  # 中文Unicode范围
                    chinese_chars += 1
        
        return chinese_chars / total_chars if total_chars > 0 else 0.0
    
    def enhanced_chinese_sentence_split(self, text: str) -> List[str]:
        """增强版中文分句"""
        # 预处理：处理特殊情况
        text = re.sub(r'([。！？；])([""''])', r'\1\n\2', text)  # 引号前换行
        text = re.sub(r'([""''])([。！？；])', r'\1\2\n', text)  # 引号后换行
        
        # 基本分句
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            if char in self.sentence_endings:
                # 检查是否在引号内
                quote_count = current_sentence.count('"') + current_sentence.count('"')
                single_quote_count = current_sentence.count(''') + current_sentence.count(''')
                
                # 如果引号成对出现，认为句子结束
                if quote_count % 2 == 0 and single_quote_count % 2 == 0:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # 处理剩余内容
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 合并过短的句子
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) < 10 and merged_sentences:
                merged_sentences[-1] += sentence
            else:
                merged_sentences.append(sentence)
        
        return [s for s in merged_sentences if s.strip()]
    
    def extract_chinese_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """提取中文关键词（词汇，词性，权重）"""
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        # 过滤有意义的词性
        meaningful_pos = {
            'n', 'nr', 'ns', 'nt', 'nw', 'nz',  # 名词类
            'v', 'vd', 'vn',                     # 动词类
            'a', 'ad', 'an',                     # 形容词类
            'l',                                 # 习语
            's',                                 # 处所词
            'f',                                 # 方位词
            'eng'                                # 英文
        }
        
        word_freq = Counter()
        word_pos = {}
        
        for word, pos in words:
            word = word.strip()
            if len(word) > 1 and pos in meaningful_pos and word not in self.chinese_punctuation:
                word_freq[word] += 1
                word_pos[word] = pos
        
        # 计算TF权重
        total_words = sum(word_freq.values())
        keyword_scores = []
        
        for word, freq in word_freq.most_common(top_k):
            tf_score = freq / total_words
            # 根据词性调整权重
            pos_weight = self._get_pos_weight(word_pos[word])
            final_score = tf_score * pos_weight
            
            keyword_scores.append((word, word_pos[word], final_score))
        
        return keyword_scores
    
    def _get_pos_weight(self, pos: str) -> float:
        """根据词性获取权重"""
        pos_weights = {
            'n': 1.2,   # 名词
            'nr': 1.5,  # 人名
            'ns': 1.3,  # 地名
            'nt': 1.4,  # 机构名
            'nw': 1.1,  # 作品名
            'nz': 1.2,  # 其他专名
            'v': 1.0,   # 动词
            'vd': 0.9,  # 副动词
            'vn': 1.1,  # 名动词
            'a': 0.8,   # 形容词
            'ad': 0.7,  # 副形词
            'an': 0.9,  # 名形词
            'l': 1.3,   # 习语
            's': 1.1,   # 处所词
            'f': 0.9,   # 方位词
            'eng': 1.2  # 英文
        }
        return pos_weights.get(pos, 1.0)
    
    def detect_dialogue_segments(self, text: str) -> List[Tuple[int, int, str]]:
        """检测对话片段 (start_pos, end_pos, dialogue_content)"""
        dialogues = []
        
        for pattern in self.dialogue_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                dialogue_content = match.group(1) if match.groups() else match.group(0)
                dialogues.append((start_pos, end_pos, dialogue_content))
        
        # 排序并去重
        dialogues = sorted(list(set(dialogues)), key=lambda x: x[0])
        return dialogues
    
    def detect_narrative_transitions(self, sentences: List[str]) -> List[int]:
        """检测叙述转换点"""
        transition_points = []
        
        for i, sentence in enumerate(sentences):
            # 检测转换词汇
            has_transition = any(word in sentence for word in self.transition_words)
            
            # 检测语篇标记
            has_discourse_marker = any(marker in sentence for marker in self.discourse_markers)
            
            # 检测时间表达
            time_patterns = [
                r'\d+年', r'\d+月', r'\d+日', r'\d+点', r'\d+分',
                r'早上', r'中午', r'下午', r'晚上', r'深夜',
                r'春天', r'夏天', r'秋天', r'冬天',
                r'昨天', r'今天', r'明天', r'后天'
            ]
            has_time_expr = any(re.search(pattern, sentence) for pattern in time_patterns)
            
            if has_transition or has_discourse_marker or has_time_expr:
                transition_points.append(i)
        
        return transition_points
    
    def calculate_chinese_semantic_features(self, text: str) -> Dict[str, float]:
        """计算中文语义特征"""
        features = {}
        
        # 中文字符比例
        features['chinese_ratio'] = self.detect_chinese_ratio(text)
        
        # 句子长度分布
        sentences = self.enhanced_chinese_sentence_split(text)
        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_std'] = np.std(sentence_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_std'] = 0
        
        # 标点密度
        punctuation_count = sum(1 for char in text if char in self.chinese_punctuation)
        features['punctuation_density'] = punctuation_count / len(text) if text else 0
        
        # 对话比例
        dialogues = self.detect_dialogue_segments(text)
        dialogue_chars = sum(end - start for start, end, _ in dialogues)
        features['dialogue_ratio'] = dialogue_chars / len(text) if text else 0
        
        # 转换词密度
        transition_count = sum(1 for word in self.transition_words if word in text)
        features['transition_density'] = transition_count / len(sentences) if sentences else 0
        
        # 词汇多样性（基于jieba分词）
        words = [word for word in jieba.cut(text) if len(word) > 1]
        unique_words = set(words)
        features['lexical_diversity'] = len(unique_words) / len(words) if words else 0
        
        return features
    
    def optimize_segmentation_for_chinese(self, sentences: List[str], 
                                        similarities: List[float]) -> List[int]:
        """针对中文优化分段边界"""
        if len(sentences) < 2:
            return []
        
        boundaries = []
        
        # 检测叙述转换点
        transition_points = self.detect_narrative_transitions(sentences)
        
        # 结合相似度和转换点
        for i in range(len(similarities)):
            similarity = similarities[i]
            
            # 基础阈值
            base_threshold = 0.5
            
            # 调整阈值
            adjusted_threshold = base_threshold
            
            # 如果是转换点，降低阈值（更容易分段）
            if i + 1 in transition_points:
                adjusted_threshold *= 0.7
            
            # 检测对话边界
            sentence_text = sentences[i] + sentences[i + 1] if i + 1 < len(sentences) else sentences[i]
            dialogues = self.detect_dialogue_segments(sentence_text)
            if dialogues:
                adjusted_threshold *= 0.8
            
            # 检测主题词变化
            current_keywords = set(word for word, _, _ in self.extract_chinese_keywords(sentences[i], 5))
            next_keywords = set(word for word, _, _ in self.extract_chinese_keywords(sentences[i + 1], 5)) if i + 1 < len(sentences) else set()
            
            keyword_overlap = len(current_keywords & next_keywords) / len(current_keywords | next_keywords) if (current_keywords | next_keywords) else 0
            
            if keyword_overlap < 0.2:  # 关键词重叠很少
                adjusted_threshold *= 0.9
            
            # 应用阈值判断
            if similarity < adjusted_threshold:
                boundaries.append(i + 1)
        
        return boundaries
    
    def post_process_chinese_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """后处理中文段落"""
        processed_paragraphs = []
        
        for paragraph in paragraphs:
            # 去除多余空白
            paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            
            # 修正标点
            paragraph = re.sub(r'\s+([。！？；，、])', r'\1', paragraph)  # 标点前不要空格
            paragraph = re.sub(r'([。！？；])([^""''\s])', r'\1 \2', paragraph)  # 句号后加空格
            
            # 修正引号
            paragraph = re.sub(r'\s*[""]\s*', '"', paragraph)  # 统一双引号
            paragraph = re.sub(r'\s*['']\s*', "'", paragraph)  # 统一单引号
            
            if paragraph:
                processed_paragraphs.append(paragraph)
        
        return processed_paragraphs


def apply_chinese_optimizations(text_processor: TextProcessor) -> TextProcessor:
    """为文本处理器应用中文优化"""
    chinese_optimizer = ChineseTextOptimizer()
    
    # 保存原始方法
    original_split_sentences = text_processor.split_sentences
    original_normalize_text = text_processor.normalize_text
    
    def enhanced_split_sentences(text: str) -> List[str]:
        """增强版分句（包含中文优化）"""
        chinese_ratio = chinese_optimizer.detect_chinese_ratio(text)
        
        if chinese_ratio > 0.5:  # 主要是中文
            return chinese_optimizer.enhanced_chinese_sentence_split(text)
        else:
            return original_split_sentences(text)
    
    def enhanced_normalize_text(text: str) -> str:
        """增强版文本标准化（包含中文优化）"""
        # 先应用原始标准化
        normalized = original_normalize_text(text)
        
        chinese_ratio = chinese_optimizer.detect_chinese_ratio(normalized)
        
        if chinese_ratio > 0.5:  # 主要是中文
            # 中文特定标准化
            # 统一中文标点
            normalized = re.sub(r'[，,]', '，', normalized)
            normalized = re.sub(r'[。.]', '。', normalized)
            normalized = re.sub(r'[！!]', '！', normalized)
            normalized = re.sub(r'[？?]', '？', normalized)
            normalized = re.sub(r'[：:]', '：', normalized)
            normalized = re.sub(r'[；;]', '；', normalized)
            
            # 处理引号
            normalized = re.sub(r'["""]', '"', normalized)
            normalized = re.sub(r"[''']", "'", normalized)
            
            # 去除中英文间多余空格
            normalized = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1 \2', normalized)
            normalized = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1 \2', normalized)
        
        return normalized
    
    # 替换方法
    text_processor.split_sentences = enhanced_split_sentences
    text_processor.normalize_text = enhanced_normalize_text
    
    # 添加中文优化器作为属性
    text_processor.chinese_optimizer = chinese_optimizer
    
    logger.info("已为文本处理器应用中文优化")
    return text_processor