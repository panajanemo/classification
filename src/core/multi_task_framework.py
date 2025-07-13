import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import os

from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextType
from src.core.transformer_segmenter import SegmentationTransformer, PositionalEncoding
from config.settings import settings


logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    TEXT_CLASSIFICATION = "text_classification"  # 文本类型分类
    BOUNDARY_DETECTION = "boundary_detection"    # 边界检测
    CONSISTENCY_PREDICTION = "consistency_prediction"  # 一致性预测


@dataclass
class TrainingExample:
    """训练样本数据结构"""
    text: str                           # 原始文本
    sentences: List[str]                # 分句结果
    text_type: TextType                 # 文本类型标签
    boundary_labels: List[int]          # 边界标签 (0/1)
    consistency_labels: List[float]     # 一致性标签 (0.0-1.0)
    metadata: Dict[str, Any]           # 元数据信息


class MultiTaskModel(nn.Module):
    """
    多任务学习模型
    联合学习文本分类、边界检测和一致性预测
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 num_text_types: int = 7,
                 num_heads: int = 6,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 shared_layers: int = 2):
        """
        初始化多任务模型
        
        Args:
            embedding_dim: 嵌入维度
            num_text_types: 文本类型数量
            num_heads: 注意力头数
            num_layers: 总Transformer层数
            dropout: Dropout比例
            shared_layers: 共享层数量
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_text_types = num_text_types
        self.shared_layers = shared_layers
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # 共享的Transformer编码器层
        shared_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(
            shared_encoder_layer, shared_layers
        )
        
        # 任务特定的Transformer层
        task_layers = num_layers - shared_layers
        if task_layers > 0:
            # 文本分类分支
            classification_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.classification_transformer = nn.TransformerEncoder(
                classification_layer, task_layers
            )
            
            # 分段任务分支
            segmentation_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.segmentation_transformer = nn.TransformerEncoder(
                segmentation_layer, task_layers
            )
        else:
            self.classification_transformer = None
            self.segmentation_transformer = None
        
        # 文本类型分类头
        self.text_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_text_types)
        )
        
        # 边界检测头
        self.boundary_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 一致性预测头
        self.consistency_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # 跨任务注意力机制
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        
        logger.info(f"多任务模型初始化: {embedding_dim}维, {num_layers}层 ({shared_layers}共享)")
    
    def forward(self, sentence_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            sentence_embeddings: 句子嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_features: 是否返回中间特征
            
        Returns:
            包含各任务预测结果的字典
        """
        batch_size, seq_len, _ = sentence_embeddings.shape
        
        # 添加位置编码
        x = self.positional_encoding(sentence_embeddings)
        
        # 处理注意力掩码
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        # 共享编码器
        shared_features = self.shared_transformer(
            x, src_key_padding_mask=key_padding_mask
        )
        
        # 任务特定编码
        if self.classification_transformer is not None:
            classification_features = self.classification_transformer(
                shared_features, src_key_padding_mask=key_padding_mask
            )
        else:
            classification_features = shared_features
            
        if self.segmentation_transformer is not None:
            segmentation_features = self.segmentation_transformer(
                shared_features, src_key_padding_mask=key_padding_mask
            )
        else:
            segmentation_features = shared_features
        
        # 跨任务信息融合
        enhanced_seg_features, _ = self.cross_task_attention(
            segmentation_features,
            classification_features,
            classification_features,
            key_padding_mask=key_padding_mask
        )
        
        # 文本类型分类 (使用全局池化)
        if attention_mask is not None:
            # 掩码平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(classification_features.size()).float()
            sum_features = torch.sum(classification_features * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            global_features = sum_features / sum_mask.clamp(min=1e-9)
        else:
            global_features = torch.mean(classification_features, dim=1)
        
        text_type_logits = self.text_classifier(global_features)
        
        # 边界检测 (对每个相邻句子对)
        boundary_predictions = []
        for i in range(seq_len - 1):
            boundary_pred = self.boundary_detector(enhanced_seg_features[:, i, :])
            boundary_predictions.append(boundary_pred)
        
        if boundary_predictions:
            boundary_predictions = torch.cat(boundary_predictions, dim=1)
        else:
            boundary_predictions = torch.zeros(batch_size, 0, device=sentence_embeddings.device)
        
        # 一致性预测 (相邻句子对的特征连接)
        consistency_predictions = []
        for i in range(seq_len - 1):
            pair_features = torch.cat([
                enhanced_seg_features[:, i, :],
                enhanced_seg_features[:, i+1, :]
            ], dim=-1)
            consistency_pred = self.consistency_predictor(pair_features)
            consistency_predictions.append(consistency_pred)
        
        if consistency_predictions:
            consistency_predictions = torch.cat(consistency_predictions, dim=1)
        else:
            consistency_predictions = torch.zeros(batch_size, 0, device=sentence_embeddings.device)
        
        results = {
            'text_type_logits': text_type_logits,
            'boundary_predictions': boundary_predictions,
            'consistency_predictions': consistency_predictions
        }
        
        if return_features:
            results.update({
                'shared_features': shared_features,
                'classification_features': classification_features,
                'segmentation_features': segmentation_features,
                'enhanced_segmentation_features': enhanced_seg_features,
                'global_features': global_features
            })
        
        return results


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 boundary_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 adaptive_weights: bool = True):
        """
        初始化多任务损失
        
        Args:
            classification_weight: 分类任务权重
            boundary_weight: 边界检测任务权重
            consistency_weight: 一致性预测任务权重
            adaptive_weights: 是否使用自适应权重
        """
        super().__init__()
        
        self.classification_weight = nn.Parameter(torch.tensor(classification_weight))
        self.boundary_weight = nn.Parameter(torch.tensor(boundary_weight))
        self.consistency_weight = nn.Parameter(torch.tensor(consistency_weight))
        self.adaptive_weights = adaptive_weights
        
        # 各任务的损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.BCELoss()
        self.consistency_loss = nn.MSELoss()
        
        # 不确定性权重（用于自适应权重）
        if adaptive_weights:
            self.log_vars = nn.Parameter(torch.zeros(3))  # 3个任务的不确定性
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            包含各项损失的字典
        """
        losses = {}
        
        # 文本分类损失
        if 'text_type_logits' in predictions and 'text_type_labels' in targets:
            cls_loss = self.classification_loss(
                predictions['text_type_logits'],
                targets['text_type_labels']
            )
            losses['classification_loss'] = cls_loss
        else:
            losses['classification_loss'] = torch.tensor(0.0)
        
        # 边界检测损失
        if 'boundary_predictions' in predictions and 'boundary_labels' in targets:
            # 确保维度匹配
            pred_boundary = predictions['boundary_predictions']
            true_boundary = targets['boundary_labels']
            
            if pred_boundary.size(1) == true_boundary.size(1):
                boundary_loss = self.boundary_loss(pred_boundary, true_boundary)
                losses['boundary_loss'] = boundary_loss
            else:
                losses['boundary_loss'] = torch.tensor(0.0)
        else:
            losses['boundary_loss'] = torch.tensor(0.0)
        
        # 一致性预测损失
        if 'consistency_predictions' in predictions and 'consistency_labels' in targets:
            pred_consistency = predictions['consistency_predictions']
            true_consistency = targets['consistency_labels']
            
            if pred_consistency.size(1) == true_consistency.size(1):
                consistency_loss = self.consistency_loss(pred_consistency, true_consistency)
                losses['consistency_loss'] = consistency_loss
            else:
                losses['consistency_loss'] = torch.tensor(0.0)
        else:
            losses['consistency_loss'] = torch.tensor(0.0)
        
        # 计算总损失
        if self.adaptive_weights and hasattr(self, 'log_vars'):
            # 使用不确定性加权 (Kendall et al., 2018)
            precision1 = torch.exp(-self.log_vars[0])
            precision2 = torch.exp(-self.log_vars[1])
            precision3 = torch.exp(-self.log_vars[2])
            
            total_loss = (
                precision1 * losses['classification_loss'] + self.log_vars[0] +
                precision2 * losses['boundary_loss'] + self.log_vars[1] +
                precision3 * losses['consistency_loss'] + self.log_vars[2]
            )
            
            losses['adaptive_weights'] = {
                'classification': precision1.item(),
                'boundary': precision2.item(),
                'consistency': precision3.item()
            }
        else:
            # 使用固定权重
            total_loss = (
                self.classification_weight * losses['classification_loss'] +
                self.boundary_weight * losses['boundary_loss'] +
                self.consistency_weight * losses['consistency_loss']
            )
        
        losses['total_loss'] = total_loss
        return losses


class MultiTaskDataset(Dataset):
    """多任务数据集"""
    
    def __init__(self, examples: List[TrainingExample], 
                 sentence_model: SentenceTransformerModel,
                 max_length: int = 128):
        """
        初始化数据集
        
        Args:
            examples: 训练样本列表
            sentence_model: 句子编码模型
            max_length: 最大序列长度
        """
        self.examples = examples
        self.sentence_model = sentence_model
        self.max_length = max_length
        
        # 文本类型到ID的映射
        self.type_to_id = {
            TextType.TECHNICAL: 0,
            TextType.NOVEL: 1,
            TextType.ACADEMIC: 2,
            TextType.NEWS: 3,
            TextType.DIALOGUE: 4,
            TextType.MIXED: 5,
            TextType.UNKNOWN: 6
        }
        
        logger.info(f"数据集初始化: {len(examples)}个样本, 最大长度: {max_length}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        example = self.examples[idx]
        
        # 编码句子
        sentences = example.sentences[:self.max_length]  # 截断
        embeddings = self.sentence_model.encode_texts(sentences)
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        seq_len = len(sentences)
        
        # 创建注意力掩码
        attention_mask = torch.ones(seq_len, dtype=torch.bool)
        
        # 处理标签
        text_type_id = self.type_to_id.get(example.text_type, 6)  # 默认为UNKNOWN
        
        # 边界标签 (确保长度匹配)
        boundary_labels = example.boundary_labels[:seq_len-1]
        if len(boundary_labels) < seq_len - 1:
            boundary_labels.extend([0] * (seq_len - 1 - len(boundary_labels)))
        
        # 一致性标签
        consistency_labels = example.consistency_labels[:seq_len-1]
        if len(consistency_labels) < seq_len - 1:
            consistency_labels.extend([0.5] * (seq_len - 1 - len(consistency_labels)))
        
        return {
            'sentence_embeddings': embeddings,
            'attention_mask': attention_mask,
            'text_type_label': torch.tensor(text_type_id, dtype=torch.long),
            'boundary_labels': torch.tensor(boundary_labels, dtype=torch.float),
            'consistency_labels': torch.tensor(consistency_labels, dtype=torch.float),
            'seq_len': seq_len
        }


class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, 
                 model: MultiTaskModel,
                 sentence_model: SentenceTransformerModel,
                 device: str = "auto",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        初始化训练器
        
        Args:
            model: 多任务模型
            sentence_model: 句子编码模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model
        self.sentence_model = sentence_model
        
        # 设备配置
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = MultiTaskLoss(adaptive_weights=True)
        
        # 训练统计
        self.train_stats = {
            'epoch': 0,
            'total_loss': [],
            'classification_loss': [],
            'boundary_loss': [],
            'consistency_loss': []
        }
        
        logger.info(f"多任务训练器初始化完成，设备: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'classification': 0.0,
            'boundary': 0.0,
            'consistency': 0.0
        }
        num_batches = 0
        
        for batch in dataloader:
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            predictions = self.model(
                batch['sentence_embeddings'],
                batch['attention_mask']
            )
            
            # 准备目标
            targets = {
                'text_type_labels': batch['text_type_label'],
                'boundary_labels': batch['boundary_labels'],
                'consistency_labels': batch['consistency_labels']
            }
            
            # 计算损失
            losses = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积损失
            epoch_losses['total'] += losses['total_loss'].item()
            epoch_losses['classification'] += losses['classification_loss'].item()
            epoch_losses['boundary'] += losses['boundary_loss'].item()
            epoch_losses['consistency'] += losses['consistency_loss'].item()
            num_batches += 1
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        # 更新统计
        self.train_stats['epoch'] += 1
        self.train_stats['total_loss'].append(avg_losses['total'])
        self.train_stats['classification_loss'].append(avg_losses['classification'])
        self.train_stats['boundary_loss'].append(avg_losses['boundary'])
        self.train_stats['consistency_loss'].append(avg_losses['consistency'])
        
        return avg_losses
    
    def save_model(self, save_path: str, include_optimizer: bool = True):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_stats': self.train_stats,
            'model_config': {
                'embedding_dim': self.model.embedding_dim,
                'num_text_types': self.model.num_text_types,
                'shared_layers': self.model.shared_layers
            }
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str, load_optimizer: bool = True):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_stats' in checkpoint:
            self.train_stats = checkpoint['train_stats']
        
        logger.info(f"模型已从 {load_path} 加载")


def create_synthetic_data(num_samples: int = 100) -> List[TrainingExample]:
    """创建合成训练数据用于测试"""
    examples = []
    
    # 示例文本模板
    templates = {
        TextType.TECHNICAL: [
            "Python是一种编程语言。它支持面向对象编程。我们可以使用类和函数。",
            "机器学习是人工智能的子领域。深度学习使用神经网络。数据预处理很重要。"
        ],
        TextType.NOVEL: [
            "他缓缓走向窗边。夕阳西下，天空泛红。她静静地看着远方。",
            "春风轻抚过脸颊。花香阵阵袭来。两人相视而笑。"
        ],
        TextType.MIXED: [
            "萧薰儿轻盈地跃上山顶。OPPO新推出的折叠屏手机很薄。科技发展日新月异。",
            "他温柔地握住她的手。今天的股市表现不错。投资需要谨慎分析。"
        ]
    }
    
    for i in range(num_samples):
        # 随机选择文本类型
        text_type = np.random.choice(list(templates.keys()))
        template = np.random.choice(templates[text_type])
        
        sentences = template.split('。')
        sentences = [s.strip() + '。' for s in sentences if s.strip()]
        
        # 生成边界标签（随机但有逻辑）
        boundary_labels = []
        for j in range(len(sentences) - 1):
            if text_type == TextType.MIXED and j == len(sentences) // 2:
                boundary_labels.append(1)  # 混合文本在中间分段
            else:
                boundary_labels.append(np.random.choice([0, 1], p=[0.7, 0.3]))
        
        # 生成一致性标签
        consistency_labels = []
        for j in range(len(sentences) - 1):
            if boundary_labels[j] == 1:
                consistency_labels.append(np.random.uniform(0.1, 0.4))  # 边界处一致性低
            else:
                consistency_labels.append(np.random.uniform(0.6, 0.9))  # 非边界处一致性高
        
        example = TrainingExample(
            text=template,
            sentences=sentences,
            text_type=text_type,
            boundary_labels=boundary_labels,
            consistency_labels=consistency_labels,
            metadata={"sample_id": i}
        )
        
        examples.append(example)
    
    return examples