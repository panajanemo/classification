import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import jieba
import re
from collections import Counter, defaultdict

from src.core.sentence_transformer_model import SentenceTransformerModel
from src.utils.text_processor import TextProcessor


logger = logging.getLogger(__name__)


class TopicConsistencyEvaluator:
    """
    主题一致性评估器
    评估文本段落间的主题连贯性和分段质量
    """
    
    def __init__(self, 
                 sentence_model: Optional[SentenceTransformerModel] = None,
                 text_processor: Optional[TextProcessor] = None,
                 language: str = "chinese"):
        """
        初始化主题一致性评估器
        
        Args:
            sentence_model: Sentence Transformer模型
            text_processor: 文本处理器
            language: 语言类型 (chinese/english)
        """
        self.sentence_model = sentence_model or SentenceTransformerModel()
        self.text_processor = text_processor or TextProcessor()
        self.language = language
        
        # 确保模型已加载
        if not self.sentence_model.is_loaded():
            self.sentence_model.load_model()
        
        # TF-IDF向量化器配置
        if language == "chinese":
            # 中文停用词
            self.stop_words = {
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', 
                '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '可以',
                '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '为什么', '但是',
                '然后', '因为', '所以', '如果', '虽然', '可是', '还是', '或者', '而且', '不过', '同时'
            }
            
            # 使用jieba分词
            def chinese_tokenizer(text):
                return [word for word in jieba.cut(text) 
                       if len(word) > 1 and word not in self.stop_words and not word.isspace()]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=chinese_tokenizer,
                lowercase=True,
                max_features=1000,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 2)
            )
        else:
            # 英文配置
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=1000,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 2)
            )
        
        # 主题模型配置
        self.lda_model = None
        self.topic_distributions = None
        
        logger.info(f"主题一致性评估器初始化完成，语言: {language}")
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """提取关键词"""
        try:
            # 使用TF-IDF提取关键词
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # 获取top-k关键词
            top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
            keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []
    
    def _calculate_semantic_similarity_matrix(self, paragraphs: List[str]) -> np.ndarray:
        """计算段落间语义相似度矩阵"""
        try:
            # 获取段落嵌入
            embeddings = self.sentence_model.encode_texts(paragraphs)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"语义相似度计算失败: {e}")
            return np.eye(len(paragraphs))  # 返回单位矩阵作为回退
    
    def _calculate_lexical_similarity_matrix(self, paragraphs: List[str]) -> np.ndarray:
        """计算段落间词汇相似度矩阵"""
        try:
            # 使用TF-IDF向量化
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraphs)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"词汇相似度计算失败: {e}")
            return np.eye(len(paragraphs))
    
    def _perform_topic_modeling(self, paragraphs: List[str], n_topics: int = None) -> Dict[str, Any]:
        """执行主题建模"""
        try:
            # 自动确定主题数
            if n_topics is None:
                n_topics = min(max(2, len(paragraphs) // 2), 8)
            
            # TF-IDF向量化
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraphs)
            
            # LDA主题建模
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100,
                learning_method='batch'
            )
            
            topic_distributions = self.lda_model.fit_transform(tfidf_matrix)
            self.topic_distributions = topic_distributions
            
            # 获取主题词
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topic_words = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[::-1][:8]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words.append(top_words)
            
            # 为每个段落分配主要主题
            paragraph_topics = []
            for i, dist in enumerate(topic_distributions):
                main_topic = np.argmax(dist)
                confidence = dist[main_topic]
                paragraph_topics.append({
                    'paragraph_id': i,
                    'main_topic': main_topic,
                    'confidence': confidence,
                    'distribution': dist.tolist()
                })
            
            return {
                'n_topics': n_topics,
                'topic_words': topic_words,
                'paragraph_topics': paragraph_topics,
                'topic_distributions': topic_distributions,
                'perplexity': self.lda_model.perplexity(tfidf_matrix) if hasattr(self.lda_model, 'perplexity') else None
            }
            
        except Exception as e:
            logger.error(f"主题建模失败: {e}")
            return {'error': str(e)}
    
    def _calculate_topic_coherence(self, paragraphs: List[str]) -> float:
        """计算主题连贯性"""
        try:
            if self.topic_distributions is None:
                return 0.0
            
            # 计算段落间主题分布的相似度
            topic_similarities = cosine_similarity(self.topic_distributions)
            
            # 计算相邻段落的主题连贯性
            coherence_scores = []
            for i in range(len(paragraphs) - 1):
                coherence_scores.append(topic_similarities[i, i + 1])
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            logger.error(f"主题连贯性计算失败: {e}")
            return 0.0
    
    def _calculate_topic_transition_smoothness(self) -> float:
        """计算主题转换平滑度"""
        try:
            if self.topic_distributions is None:
                return 0.0
            
            transition_scores = []
            for i in range(len(self.topic_distributions) - 1):
                # 计算主题分布变化的JS散度
                p = self.topic_distributions[i]
                q = self.topic_distributions[i + 1]
                
                # 添加小量避免log(0)
                p = p + 1e-10
                q = q + 1e-10
                
                # 计算KL散度
                m = (p + q) / 2
                js_div = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
                
                # 转换为相似度 (0-1)
                similarity = np.exp(-js_div)
                transition_scores.append(similarity)
            
            return np.mean(transition_scores) if transition_scores else 0.0
            
        except Exception as e:
            logger.error(f"主题转换平滑度计算失败: {e}")
            return 0.0
    
    def _detect_topic_boundaries(self, paragraphs: List[str]) -> List[int]:
        """检测主题边界"""
        try:
            if self.topic_distributions is None:
                return []
            
            boundaries = []
            threshold = 0.3  # 主题变化阈值
            
            for i in range(len(self.topic_distributions) - 1):
                # 计算相邻段落的主题相似度
                similarity = cosine_similarity(
                    [self.topic_distributions[i]], 
                    [self.topic_distributions[i + 1]]
                )[0, 0]
                
                # 如果相似度低于阈值，认为存在主题边界
                if similarity < threshold:
                    boundaries.append(i + 1)
            
            return boundaries
            
        except Exception as e:
            logger.error(f"主题边界检测失败: {e}")
            return []
    
    def evaluate_segmentation_consistency(self, paragraphs: List[str]) -> Dict[str, Any]:
        """
        评估分段的主题一致性
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            一致性评估结果
        """
        if not paragraphs or len(paragraphs) < 2:
            return {"error": "段落数量不足，无法评估一致性"}
        
        try:
            # 1. 语义相似度分析
            semantic_similarity_matrix = self._calculate_semantic_similarity_matrix(paragraphs)
            avg_semantic_similarity = np.mean(semantic_similarity_matrix[np.triu_indices_from(semantic_similarity_matrix, k=1)])
            
            # 2. 词汇相似度分析
            lexical_similarity_matrix = self._calculate_lexical_similarity_matrix(paragraphs)
            avg_lexical_similarity = np.mean(lexical_similarity_matrix[np.triu_indices_from(lexical_similarity_matrix, k=1)])
            
            # 3. 主题建模分析
            topic_analysis = self._perform_topic_modeling(paragraphs)
            
            # 4. 主题连贯性评估
            topic_coherence = self._calculate_topic_coherence(paragraphs)
            topic_smoothness = self._calculate_topic_transition_smoothness()
            
            # 5. 主题边界检测
            detected_boundaries = self._detect_topic_boundaries(paragraphs)
            
            # 6. 段落内一致性评估
            paragraph_consistencies = []
            for i, paragraph in enumerate(paragraphs):
                sentences = self.text_processor.split_sentences(paragraph)
                if len(sentences) > 1:
                    # 计算段落内句子的一致性
                    sentence_embeddings = self.sentence_model.encode_texts(sentences)
                    sentence_similarities = cosine_similarity(sentence_embeddings)
                    internal_consistency = np.mean(sentence_similarities[np.triu_indices_from(sentence_similarities, k=1)])
                    paragraph_consistencies.append(internal_consistency)
                else:
                    paragraph_consistencies.append(1.0)  # 单句段落一致性为1
            
            # 7. 关键词重叠分析
            paragraph_keywords = []
            for paragraph in paragraphs:
                keywords = self._extract_keywords(paragraph, top_k=5)
                paragraph_keywords.append([kw[0] for kw in keywords])
            
            # 计算相邻段落的关键词重叠率
            keyword_overlaps = []
            for i in range(len(paragraph_keywords) - 1):
                current_kw = set(paragraph_keywords[i])
                next_kw = set(paragraph_keywords[i + 1])
                if current_kw or next_kw:
                    overlap = len(current_kw & next_kw) / len(current_kw | next_kw)
                    keyword_overlaps.append(overlap)
                else:
                    keyword_overlaps.append(0.0)
            
            # 8. 综合一致性分数计算
            consistency_score = self._calculate_overall_consistency_score(
                avg_semantic_similarity, avg_lexical_similarity, topic_coherence,
                topic_smoothness, np.mean(paragraph_consistencies),
                np.mean(keyword_overlaps) if keyword_overlaps else 0.0
            )
            
            return {
                'consistency_score': round(consistency_score, 3),
                'semantic_analysis': {
                    'avg_similarity': round(avg_semantic_similarity, 3),
                    'similarity_matrix': semantic_similarity_matrix.tolist()
                },
                'lexical_analysis': {
                    'avg_similarity': round(avg_lexical_similarity, 3),
                    'similarity_matrix': lexical_similarity_matrix.tolist()
                },
                'topic_analysis': topic_analysis,
                'coherence_metrics': {
                    'topic_coherence': round(topic_coherence, 3),
                    'topic_smoothness': round(topic_smoothness, 3),
                    'detected_boundaries': detected_boundaries
                },
                'paragraph_metrics': {
                    'internal_consistencies': [round(c, 3) for c in paragraph_consistencies],
                    'avg_internal_consistency': round(np.mean(paragraph_consistencies), 3),
                    'keyword_overlaps': [round(o, 3) for o in keyword_overlaps],
                    'avg_keyword_overlap': round(np.mean(keyword_overlaps) if keyword_overlaps else 0.0, 3)
                },
                'keywords_by_paragraph': paragraph_keywords,
                'evaluation_summary': {
                    'paragraph_count': len(paragraphs),
                    'avg_paragraph_length': round(np.mean([len(p) for p in paragraphs]), 2),
                    'topic_boundary_detected': len(detected_boundaries),
                    'overall_consistency': 'high' if consistency_score > 0.7 else 'medium' if consistency_score > 0.4 else 'low'
                }
            }
            
        except Exception as e:
            logger.error(f"一致性评估失败: {e}")
            return {"error": f"评估处理错误: {str(e)}"}
    
    def _calculate_overall_consistency_score(self, semantic_sim: float, lexical_sim: float,
                                           topic_coherence: float, topic_smoothness: float,
                                           internal_consistency: float, keyword_overlap: float) -> float:
        """计算综合一致性分数"""
        # 加权组合各项指标
        weights = {
            'semantic': 0.3,      # 语义相似度权重
            'lexical': 0.2,       # 词汇相似度权重
            'topic_coherence': 0.2,  # 主题连贯性权重
            'topic_smoothness': 0.15, # 主题平滑度权重
            'internal': 0.1,      # 段落内一致性权重
            'keyword': 0.05       # 关键词重叠权重
        }
        
        score = (
            weights['semantic'] * semantic_sim +
            weights['lexical'] * lexical_sim +
            weights['topic_coherence'] * topic_coherence +
            weights['topic_smoothness'] * topic_smoothness +
            weights['internal'] * internal_consistency +
            weights['keyword'] * keyword_overlap
        )
        
        return max(0.0, min(1.0, score))
    
    def compare_segmentation_consistency(self, 
                                       segmentation_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        比较不同分段方法的一致性
        
        Args:
            segmentation_results: 不同方法的分段结果 {method_name: paragraphs}
            
        Returns:
            比较结果
        """
        comparison_results = {}
        
        for method_name, paragraphs in segmentation_results.items():
            try:
                consistency_result = self.evaluate_segmentation_consistency(paragraphs)
                comparison_results[method_name] = consistency_result
            except Exception as e:
                logger.error(f"方法 {method_name} 一致性评估失败: {e}")
                comparison_results[method_name] = {"error": str(e)}
        
        # 生成比较摘要
        valid_results = {k: v for k, v in comparison_results.items() if "error" not in v}
        
        if valid_results:
            # 找出最佳方法
            consistency_scores = {k: v['consistency_score'] for k, v in valid_results.items()}
            best_method = max(consistency_scores, key=consistency_scores.get)
            
            comparison_summary = {
                'best_method': best_method,
                'consistency_ranking': sorted(consistency_scores.items(), key=lambda x: x[1], reverse=True),
                'score_differences': {
                    method: round(consistency_scores[best_method] - score, 3)
                    for method, score in consistency_scores.items()
                },
                'methods_compared': len(valid_results)
            }
        else:
            comparison_summary = {'error': '所有方法的一致性评估都失败了'}
        
        return {
            'detailed_results': comparison_results,
            'comparison_summary': comparison_summary
        }