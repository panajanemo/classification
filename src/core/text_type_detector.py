import re
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
import jieba


logger = logging.getLogger(__name__)


class TextType(Enum):
    """文本类型枚举"""
    TECHNICAL = "technical"      # 技术文档
    NOVEL = "novel"             # 小说文学
    NEWS = "news"               # 新闻报道
    ACADEMIC = "academic"       # 学术论文
    DIALOGUE = "dialogue"       # 对话文本
    MIXED = "mixed"             # 混合类型
    UNKNOWN = "unknown"         # 未知类型


class TextTypeDetector:
    """文本类型检测器"""
    
    def __init__(self):
        """初始化文本类型检测器"""
        
        # 技术文档特征词汇
        self.technical_keywords = {
            "函数", "方法", "类", "接口", "API", "算法", "数据结构", "数据库", "服务器",
            "配置", "部署", "测试", "代码", "编程", "开发", "框架", "库", "模块",
            "Python", "Java", "JavaScript", "C++", "HTML", "CSS", "SQL", "Git",
            "Docker", "Kubernetes", "Linux", "系统", "网络", "安全", "性能",
            "def", "class", "import", "function", "method", "object", "string",
            "array", "list", "dict", "loop", "condition", "variable"
        }
        
        # 小说文学特征词汇（大幅扩展）
        self.novel_keywords = {
            # 描述性词汇
            "心中", "眼中", "脸上", "手中", "身边", "面前", "背后", "心里", "眼里", "怀中",
            "耳边", "唇边", "眉头", "眼角", "嘴角", "指尖", "掌心", "胸口", "肩膀",
            
            # 情感表达
            "笑容", "微笑", "苦笑", "冷笑", "轻笑", "大笑", "皱眉", "蹙眉", "点头", "摇头", 
            "叹息", "叹气", "深呼吸", "沉默", "无语", "哽咽", "流泪", "哭泣",
            
            # 情绪形容
            "温柔", "冷漠", "愤怒", "喜悦", "悲伤", "恐惧", "惊讶", "疑惑", "紧张", "兴奋",
            "失望", "绝望", "希望", "欣喜", "忧伤", "痛苦", "快乐", "满足", "不安", "焦虑",
            
            # 外貌描述
            "美丽", "漂亮", "英俊", "帅气", "可爱", "清秀", "俊美", "绝色", "倾城", "出众",
            "优雅", "高贵", "神秘", "魅力", "迷人", "动人", "醉人", "惊艳",
            
            # 性格特征
            "强大", "弱小", "勇敢", "胆怯", "聪明", "愚蠢", "善良", "邪恶", "单纯", "复杂",
            "天真", "成熟", "冷静", "冲动", "理智", "感性", "坚强", "脆弱",
            
            # 对话标识
            "她说", "他说", "我说", "说道", "道", "回答", "询问", "低语", "轻声", "大声",
            "喃喃", "呢喃", "嘀咕", "嘟囔", "开口", "启唇", "张嘴", "闭嘴", "言道",
            
            # 时间副词
            "突然", "忽然", "瞬间", "片刻", "许久", "良久", "半晌", "刹那", "顷刻", "霎时",
            "一瞬", "转眼", "眨眼", "不久", "稍后", "随即", "接着", "然后", "紧接着",
            
            # 比喻词汇
            "宛如", "仿佛", "好似", "犹如", "就像", "似乎", "看起来", "听起来", "感觉像",
            "如同", "恰似", "正如", "好比", "宛若", "形如", "状如",
            
            # 小说特有动作
            "凝视", "注视", "端详", "打量", "扫视", "环顾", "回眸", "侧目", "瞥见", "窥视",
            "踱步", "漫步", "疾走", "奔跑", "冲向", "走向", "靠近", "远离", "转身", "回头",
            "伸手", "抬手", "挥手", "摆手", "握拳", "松手", "抚摸", "轻抚", "拥抱", "推开",
            
            # 心理活动
            "思考", "思索", "沉思", "深思", "琢磨", "考虑", "犹豫", "迟疑", "决定", "下定决心",
            "恍然", "明白", "理解", "困惑", "疑惑", "担心", "担忧", "放心", "安心",
            
            # 场景描述
            "房间", "卧室", "客厅", "书房", "厨房", "花园", "庭院", "阳台", "窗边", "门口",
            "街道", "小巷", "广场", "公园", "咖啡厅", "餐厅", "学校", "办公室",
            
            # 网络小说特征
            "系统", "任务", "升级", "等级", "经验", "技能", "属性", "装备", "道具", "背包",
            "修为", "境界", "功法", "武技", "灵力", "法力", "真气", "内力", "丹药", "宝物",
            "师父", "师兄", "师姐", "长老", "宗主", "掌门", "弟子", "门派", "宗门", "家族"
        }
        
        # 学术论文特征词汇
        self.academic_keywords = {
            "研究", "分析", "实验", "方法", "结果", "结论", "讨论", "理论", "模型",
            "假设", "验证", "数据", "统计", "显著", "相关", "影响", "因素",
            "参考文献", "摘要", "关键词", "引言", "方法论", "实证", "定量", "定性",
            "样本", "变量", "指标", "测量", "评估", "比较", "对比", "差异",
            "因此", "所以", "由于", "基于", "根据", "通过", "采用", "应用"
        }
        
        # 新闻报道特征词汇
        self.news_keywords = {
            "记者", "报道", "采访", "发布", "宣布", "举行", "召开", "出席", "参加",
            "政府", "部门", "官员", "领导", "市民", "群众", "社会", "公众",
            "今天", "昨天", "明天", "今年", "去年", "明年", "近日", "日前",
            "据", "表示", "指出", "强调", "透露", "介绍", "说明", "解释",
            "消息", "新闻", "事件", "情况", "问题", "措施", "政策", "决定"
        }
        
        # 对话文本特征模式
        self.dialogue_patterns = [
            r'["""].*?["""]',  # 引号对话
            r'「.*?」',         # 日式引号
            r'『.*?』',         # 中式书名号（常用于对话）
            r'^\s*[A-Z\u4e00-\u9fa5]+\s*[：:]\s*',  # 姓名：对话格式
            r'^\s*\d+\.\s*[A-Z\u4e00-\u9fa5]+\s*[：:]\s*'  # 编号.姓名：对话格式
        ]
        
        logger.info("文本类型检测器初始化完成")
    
    def _count_keyword_matches(self, text: str, keywords: set) -> int:
        """计算关键词匹配数量"""
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                count += 1
        return count
    
    def _detect_code_patterns(self, text: str) -> int:
        """检测代码模式"""
        code_patterns = [
            r'def\s+\w+\s*\(',     # Python函数定义
            r'class\s+\w+\s*\(',   # Python类定义
            r'import\s+\w+',       # import语句
            r'from\s+\w+\s+import', # from import语句
            r'```[a-zA-Z]*\n',     # 代码块标记
            r'`[^`]+`',            # 行内代码
            r'\w+\.\w+\(',         # 方法调用
            r'=\s*["\'][^"\']*["\']', # 字符串赋值
            r'{\s*\w+\s*:\s*\w+\s*}', # 字典格式
            r'\[\s*\w+\s*,\s*\w+\s*\]' # 列表格式
        ]
        
        count = 0
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        
        return count
    
    def _detect_dialogue_patterns(self, text: str) -> int:
        """检测对话模式"""
        count = 0
        for pattern in self.dialogue_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            count += len(matches)
        
        # 检测引号数量
        quote_count = text.count('"') + text.count('"') + text.count('"')
        count += quote_count // 2  # 成对的引号
        
        return count
    
    def _detect_novel_patterns(self, text: str) -> int:
        """检测小说特有模式"""
        count = 0
        
        # 心理描写模式
        psychological_patterns = [
            r'心[中里]想[着道]?',         # 心中想着/心里想道
            r'[他她我]的心[中里]',         # 他的心中
            r'暗自[想说]',                # 暗自想/暗自说
            r'[想觉]得',                  # 想得/觉得
            r'内心[的]?[想说]',           # 内心想/内心说
        ]
        
        # 动作描写模式
        action_patterns = [
            r'[他她我][轻慢急缓]?[走跑]',    # 他走/她慢跑
            r'[转回]身[向朝]?',             # 转身向/回身朝
            r'[伸抬挥]手',                  # 伸手/抬手/挥手
            r'[点摇][了着]?头',             # 点头/摇头
            r'[皱蹙][了着]?眉',             # 皱眉/蹙眉
            r'[闭合睁]着?眼',               # 闭眼/合眼/睁眼
        ]
        
        # 表情描写模式
        expression_patterns = [
            r'[微苦冷轻][笑道说]',          # 微笑/苦笑道
            r'脸上[露浮现带][出着]',        # 脸上露出/脸上浮现
            r'眼[中里]闪[过着烁]',          # 眼中闪过/眼里闪烁
            r'[温冷]柔[的]?[笑容声音]',     # 温柔的笑容/冷柔的声音
        ]
        
        # 时间转换模式
        time_patterns = [
            r'[突忽][然而]',                # 突然/忽然
            r'[片瞬刹][刻间那]',            # 片刻/瞬间/刹那
            r'[不一过]久',                  # 不久/一久/过久
            r'[随紧接]即',                  # 随即/紧接着
        ]
        
        all_patterns = psychological_patterns + action_patterns + expression_patterns + time_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        
        # 检测人称代词的使用频率（小说特征）
        pronoun_pattern = r'[他她我][们]?[的]?'
        pronoun_matches = re.findall(pronoun_pattern, text)
        count += len(pronoun_matches) // 10  # 适度加权
        
        return count
    
    def _analyze_sentence_structure(self, text: str) -> Dict[str, float]:
        """分析句子结构特征"""
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"avg_length": 0, "complexity": 0, "question_ratio": 0}
        
        # 平均句子长度
        total_length = sum(len(s) for s in sentences)
        avg_length = total_length / len(sentences)
        
        # 复杂度（逗号密度）
        comma_count = text.count('，') + text.count(',')
        complexity = comma_count / len(sentences) if sentences else 0
        
        # 疑问句比例
        question_count = text.count('？') + text.count('?')
        question_ratio = question_count / len(sentences) if sentences else 0
        
        return {
            "avg_length": avg_length,
            "complexity": complexity,
            "question_ratio": question_ratio
        }
    
    def detect_text_type(self, text: str) -> Tuple[TextType, Dict[str, float]]:
        """
        检测文本类型
        
        Args:
            text: 输入文本
            
        Returns:
            (文本类型, 置信度得分)
        """
        if not text or len(text.strip()) < 50:
            return TextType.UNKNOWN, {"confidence": 0.0}
        
        # 计算各类型特征得分
        scores = {}
        
        # 技术文档得分
        tech_keywords = self._count_keyword_matches(text, self.technical_keywords)
        code_patterns = self._detect_code_patterns(text)
        scores["technical"] = (tech_keywords * 2 + code_patterns * 3) / len(text) * 1000
        
        # 小说文学得分（增强检测）
        novel_keywords = self._count_keyword_matches(text, self.novel_keywords)
        dialogue_patterns = self._detect_dialogue_patterns(text)
        novel_patterns = self._detect_novel_patterns(text)
        scores["novel"] = (novel_keywords * 1.5 + dialogue_patterns * 1.5 + novel_patterns * 2) / len(text) * 1000
        
        # 学术论文得分
        academic_keywords = self._count_keyword_matches(text, self.academic_keywords)
        structure = self._analyze_sentence_structure(text)
        scores["academic"] = (academic_keywords * 2 + structure["complexity"] * 5) / len(text) * 1000
        
        # 新闻报道得分
        news_keywords = self._count_keyword_matches(text, self.news_keywords)
        scores["news"] = news_keywords / len(text) * 1000
        
        # 对话文本得分
        scores["dialogue"] = dialogue_patterns / len(text) * 1000
        
        # 找到最高得分的类型
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # 置信度计算
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0
        
        # 根据得分和置信度确定类型
        if max_score < 1.0:  # 得分太低
            detected_type = TextType.UNKNOWN
        elif confidence < 0.3:  # 置信度太低，可能是混合类型
            detected_type = TextType.MIXED
        else:
            type_mapping = {
                "technical": TextType.TECHNICAL,
                "novel": TextType.NOVEL,
                "academic": TextType.ACADEMIC,
                "news": TextType.NEWS,
                "dialogue": TextType.DIALOGUE
            }
            detected_type = type_mapping.get(max_type, TextType.UNKNOWN)
        
        # 准备详细得分信息
        detailed_scores = {
            "confidence": confidence,
            "type_scores": scores,
            "max_score": max_score,
            "structure_info": structure
        }
        
        logger.debug(f"检测到文本类型: {detected_type.value}, 置信度: {confidence:.3f}")
        
        return detected_type, detailed_scores
    
    def get_segmentation_config(self, text_type: TextType) -> Dict[str, float]:
        """
        根据文本类型获取分段配置
        
        Args:
            text_type: 文本类型
            
        Returns:
            分段配置参数
        """
        configs = {
            TextType.TECHNICAL: {
                "threshold": 0.4,           # 技术文档需要更细的分段
                "window_size": 2,           # 较小的窗口
                "min_paragraph_length": 100,
                "max_paragraph_length": 800,
                "use_structure_hints": True  # 利用结构提示（如标题、代码块）
            },
            TextType.NOVEL: {
                "threshold": 0.6,           # 小说保持较大的段落
                "window_size": 4,           # 较大的窗口
                "min_paragraph_length": 200,
                "max_paragraph_length": 1000,
                "use_structure_hints": False
            },
            TextType.ACADEMIC: {
                "threshold": 0.45,          # 学术文档中等分段
                "window_size": 3,
                "min_paragraph_length": 150,
                "max_paragraph_length": 600,
                "use_structure_hints": True
            },
            TextType.NEWS: {
                "threshold": 0.5,           # 新闻标准分段
                "window_size": 3,
                "min_paragraph_length": 80,
                "max_paragraph_length": 400,
                "use_structure_hints": False
            },
            TextType.DIALOGUE: {
                "threshold": 0.3,           # 对话需要很细的分段
                "window_size": 2,
                "min_paragraph_length": 50,
                "max_paragraph_length": 300,
                "use_structure_hints": False
            },
            TextType.MIXED: {
                "threshold": 0.5,           # 混合类型使用默认配置
                "window_size": 3,
                "min_paragraph_length": 100,
                "max_paragraph_length": 500,
                "use_structure_hints": False
            },
            TextType.UNKNOWN: {
                "threshold": 0.5,           # 未知类型使用默认配置
                "window_size": 3,
                "min_paragraph_length": 100,
                "max_paragraph_length": 500,
                "use_structure_hints": False
            }
        }
        
        return configs.get(text_type, configs[TextType.UNKNOWN])