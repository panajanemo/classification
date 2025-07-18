from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union


class SegmentationRequest(BaseModel):
    """语义分段请求模型"""
    
    text: str = Field(..., description="待分段的文本", min_length=1, max_length=10000)
    threshold: Optional[float] = Field(
        None, 
        description="分段阈值(0.0-1.0)，值越小分段越细", 
        ge=0.0, 
        le=1.0
    )
    separator: Optional[str] = Field(
        None, 
        description="段落间分隔符", 
        max_length=10
    )
    
    @validator('text')
    def validate_text(cls, v):
        """验证文本内容"""
        if not v or not v.strip():
            raise ValueError("文本内容不能为空")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "这是第一句话。这是第二句话，内容相关。这是第三句话，讨论不同主题。这是第四句话，又回到第二句的主题。",
                "threshold": 0.5,
                "separator": "\\n\\n"
            }
        }


class QualityMetrics(BaseModel):
    """分段质量指标模型"""
    
    paragraph_count: int = Field(..., description="段落数量")
    avg_length: float = Field(..., description="平均段落长度")
    length_std: float = Field(..., description="段落长度标准差")
    too_short_count: int = Field(..., description="过短段落数量")
    too_long_count: int = Field(..., description="过长段落数量")
    semantic_consistency: float = Field(..., description="语义一致性分数")
    quality_score: float = Field(..., description="综合质量分数")


class SegmentationResponse(BaseModel):
    """语义分段响应模型"""
    
    success: bool = Field(..., description="处理是否成功")
    paragraphs: List[str] = Field(..., description="分段结果段落列表")
    formatted_text: str = Field(..., description="格式化后的完整文本")
    sentence_count: int = Field(..., description="原文句子数量")
    boundary_count: int = Field(..., description="段落边界数量")
    quality: QualityMetrics = Field(..., description="分段质量指标")
    processing_time: float = Field(..., description="处理耗时(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "paragraphs": [
                    "这是第一段，包含相关的句子内容。",
                    "这是第二段，讨论不同的主题内容。"
                ],
                "formatted_text": "这是第一段，包含相关的句子内容。\\n\\n这是第二段，讨论不同的主题内容。",
                "sentence_count": 4,
                "boundary_count": 1,
                "quality": {
                    "paragraph_count": 2,
                    "avg_length": 45.5,
                    "length_std": 2.5,
                    "too_short_count": 0,
                    "too_long_count": 0,
                    "semantic_consistency": 0.75,
                    "quality_score": 0.85
                },
                "processing_time": 0.32
            }
        }


class ErrorResponse(BaseModel):
    """错误响应模型"""
    
    success: bool = Field(False, description="处理失败")
    error: str = Field(..., description="错误信息")
    error_type: str = Field(..., description="错误类型")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "输入文本过长",
                "error_type": "ValidationError",
                "details": {
                    "max_length": 10000,
                    "actual_length": 15000
                }
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    model_info: Dict[str, Any] = Field(..., description="模型信息")
    uptime: float = Field(..., description="运行时间(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "model_info": {
                    "model_name": "dennlinger/bert-wiki-paragraphs",
                    "device": "cpu",
                    "is_loaded": True
                },
                "uptime": 3600.5
            }
        }


class ConfigRequest(BaseModel):
    """配置更新请求模型"""
    
    threshold: Optional[float] = Field(
        None, 
        description="新的分段阈值", 
        ge=0.0, 
        le=1.0
    )
    min_paragraph_length: Optional[int] = Field(
        None, 
        description="最小段落长度", 
        ge=10
    )
    max_paragraph_length: Optional[int] = Field(
        None, 
        description="最大段落长度", 
        ge=100
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "threshold": 0.6,
                "min_paragraph_length": 60,
                "max_paragraph_length": 400
            }
        }


class ConfigResponse(BaseModel):
    """配置响应模型"""
    
    success: bool = Field(..., description="配置是否成功")
    current_config: Dict[str, Any] = Field(..., description="当前配置")
    message: str = Field(..., description="操作消息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "current_config": {
                    "threshold": 0.6,
                    "min_paragraph_length": 60,
                    "max_paragraph_length": 400
                },
                "message": "配置更新成功"
            }
        }


class MarkdownUploadRequest(BaseModel):
    """Markdown文件上传请求模型"""
    
    threshold: Optional[float] = Field(
        0.5, 
        description="分段阈值(0.0-1.0)，值越小分段越细", 
        ge=0.0, 
        le=1.0
    )
    preserve_structure: bool = Field(
        True, 
        description="是否保持Markdown结构"
    )
    chunk_size: Optional[int] = Field(
        5000, 
        description="分块处理大小", 
        ge=1000, 
        le=20000
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "threshold": 0.5,
                "preserve_structure": True,
                "chunk_size": 5000
            }
        }


class MarkdownProcessingStats(BaseModel):
    """Markdown处理统计信息模型"""
    
    total_blocks: int = Field(..., description="总块数")
    block_types: Dict[str, int] = Field(..., description="块类型统计")
    headings: Dict[str, Any] = Field(..., description="标题统计")
    paragraphs: Dict[str, Any] = Field(..., description="段落统计")
    code_blocks: int = Field(..., description="代码块数量")
    lists: int = Field(..., description="列表数量")
    tables: int = Field(..., description="表格数量")


class MarkdownProcessingResponse(BaseModel):
    """Markdown处理响应模型"""
    
    success: bool = Field(..., description="处理是否成功")
    filename: str = Field(..., description="原文件名")
    file_size: int = Field(..., description="文件大小(字节)")
    processed_markdown: str = Field(..., description="处理后的Markdown内容")
    original_stats: MarkdownProcessingStats = Field(..., description="原始文档统计")
    processed_stats: MarkdownProcessingStats = Field(..., description="处理后文档统计")
    blocks_count: Dict[str, int] = Field(..., description="块数量变化")
    processing_info: Dict[str, Any] = Field(..., description="处理信息")
    processing_time: float = Field(..., description="处理耗时(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "document.md",
                "file_size": 15420,
                "processed_markdown": "# 标题\n\n段落内容...",
                "original_stats": {
                    "total_blocks": 5,
                    "block_types": {"paragraph": 3, "heading": 2},
                    "headings": {"count": 2, "levels": {1: 1, 2: 1}},
                    "paragraphs": {"count": 3, "total_chars": 500, "avg_length": 166.7},
                    "code_blocks": 0,
                    "lists": 0,
                    "tables": 0
                },
                "processed_stats": {
                    "total_blocks": 7,
                    "block_types": {"paragraph": 5, "heading": 2},
                    "headings": {"count": 2, "levels": {1: 1, 2: 1}},
                    "paragraphs": {"count": 5, "total_chars": 500, "avg_length": 100.0},
                    "code_blocks": 0,
                    "lists": 0,
                    "tables": 0
                },
                "blocks_count": {"original": 5, "processed": 7},
                "processing_info": {
                    "threshold": 0.5,
                    "segmentation_applied": True
                },
                "processing_time": 2.34
            }
        }


class FileUploadErrorResponse(BaseModel):
    """文件上传错误响应模型"""
    
    success: bool = Field(False, description="处理失败")
    error: str = Field(..., description="错误信息")
    error_type: str = Field(..., description="错误类型")
    filename: Optional[str] = Field(None, description="文件名")
    file_size: Optional[int] = Field(None, description="文件大小")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "文件类型不支持",
                "error_type": "ValidationError",
                "filename": "document.txt",
                "file_size": 1024,
                "details": {
                    "supported_types": [".md", ".markdown"],
                    "received_type": ".txt"
                }
            }
        }


class ProcessingStatusResponse(BaseModel):
    """处理状态响应模型"""
    
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="处理状态")
    progress: float = Field(..., description="处理进度(0-100)")
    message: str = Field(..., description="状态消息")
    estimated_time: Optional[float] = Field(None, description="预估剩余时间(秒)")
    result_available: bool = Field(False, description="结果是否可用")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "md_proc_123456",
                "status": "processing",
                "progress": 65.5,
                "message": "正在处理第3章节...",
                "estimated_time": 45.2,
                "result_available": False
            }
        }


# 增强版分段相关模型

class EnhancedSegmentationRequest(BaseModel):
    """增强版语义分段请求模型"""
    
    text: str = Field(..., description="待分段的文本", min_length=1, max_length=20000)
    threshold: Optional[float] = Field(
        None, 
        description="基础分段阈值(0.0-1.0)，会根据文本类型自动调整", 
        ge=0.0, 
        le=1.0
    )
    enable_auto_threshold: Optional[bool] = Field(
        None,
        description="是否启用自动阈值调整"
    )
    enable_structure_hints: Optional[bool] = Field(
        None,
        description="是否启用结构提示检测(如标题、列表等)"
    )
    enable_hierarchical_output: Optional[bool] = Field(
        None,
        description="是否启用层次化输出格式"
    )
    force_text_type: Optional[str] = Field(
        None,
        description="强制指定文本类型: technical/novel/academic/news/dialogue"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """验证文本内容"""
        if not v or not v.strip():
            raise ValueError("文本内容不能为空")
        return v.strip()
    
    @validator('force_text_type')
    def validate_text_type(cls, v):
        """验证文本类型"""
        if v is not None:
            valid_types = ['technical', 'novel', 'academic', 'news', 'dialogue', 'mixed']
            if v not in valid_types:
                raise ValueError(f"文本类型必须是以下之一: {', '.join(valid_types)}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "人工智能技术发展迅速。机器学习是其核心分支。深度学习通过神经网络进行学习。Python是常用的编程语言。它具有简洁的语法特点。",
                "threshold": 0.5,
                "enable_auto_threshold": True,
                "enable_structure_hints": True,
                "enable_hierarchical_output": True,
                "force_text_type": None
            }
        }


class HierarchicalParagraph(BaseModel):
    """层次化段落模型"""
    
    text: str = Field(..., description="段落文本内容")
    type: str = Field(..., description="段落类型: content/title/list_item")
    level: int = Field(..., description="层次级别")
    length: int = Field(..., description="段落字符长度")
    sentence_count: int = Field(..., description="包含的句子数量")
    start_idx: int = Field(..., description="起始句子索引")
    end_idx: int = Field(..., description="结束句子索引")
    key_phrases: List[str] = Field(..., description="关键短语列表")


class EnhancedQualityMetrics(BaseModel):
    """增强版分段质量指标模型"""
    
    paragraph_count: int = Field(..., description="段落数量")
    avg_length: float = Field(..., description="平均段落长度")
    length_std: float = Field(..., description="段落长度标准差")
    avg_sentence_count: float = Field(..., description="平均句子数量")
    too_short_count: int = Field(..., description="过短段落数量")
    too_long_count: int = Field(..., description="过长段落数量")
    semantic_consistency: float = Field(..., description="语义一致性分数")
    structure_score: float = Field(..., description="结构合理性分数")
    type_distribution: Dict[str, int] = Field(..., description="段落类型分布")
    quality_score: float = Field(..., description="综合质量分数")


class EnhancedSegmentationResponse(BaseModel):
    """增强版语义分段响应模型"""
    
    success: bool = Field(True, description="处理是否成功")
    paragraphs: List[HierarchicalParagraph] = Field(..., description="层次化段落列表")
    text_type: str = Field(..., description="检测到的文本类型")
    type_confidence: float = Field(..., description="文本类型检测置信度")
    config_used: Dict[str, Any] = Field(..., description="实际使用的配置")
    sentence_count: int = Field(..., description="原始句子数量")
    boundary_count: int = Field(..., description="检测到的边界数量")
    multi_scale_info: Dict[str, int] = Field(..., description="多尺度分析信息")
    quality: EnhancedQualityMetrics = Field(..., description="分段质量评估")
    processing_time: float = Field(..., description="处理时间(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "paragraphs": [
                    {
                        "text": "人工智能技术发展迅速。机器学习是其核心分支。",
                        "type": "content",
                        "level": 1,
                        "length": 24,
                        "sentence_count": 2,
                        "start_idx": 0,
                        "end_idx": 2,
                        "key_phrases": ["人工智能", "技术", "机器学习"]
                    }
                ],
                "text_type": "technical",
                "type_confidence": 0.85,
                "config_used": {"threshold": 0.4, "window_size": 2},
                "sentence_count": 5,
                "boundary_count": 2,
                "multi_scale_info": {"scale_1": 4, "scale_3": 3, "scale_5": 1},
                "quality": {
                    "paragraph_count": 2,
                    "avg_length": 45.5,
                    "length_std": 12.3,
                    "avg_sentence_count": 2.5,
                    "too_short_count": 0,
                    "too_long_count": 0,
                    "semantic_consistency": 0.75,
                    "structure_score": 0.90,
                    "type_distribution": {"content": 2},
                    "quality_score": 0.82
                },
                "processing_time": 0.245
            }
        }