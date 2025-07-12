from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """应用配置"""
    
    # 服务配置
    app_name: str = "BERT语义分段服务"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # 模型配置
    model_name: str = "dennlinger/bert-wiki-paragraphs"  # 原BERT模型
    enhanced_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 增强模型
    model_cache_dir: Optional[str] = None
    max_length: int = 512
    use_enhanced_segmenter: bool = True  # 是否使用增强版分段器
    
    # 分段配置
    threshold: float = 0.5
    min_paragraph_length: int = 50
    max_paragraph_length: int = 500
    separator: str = "\n\n"
    
    # 增强分段配置
    enable_auto_threshold: bool = True  # 启用自动阈值调整
    multi_scale_windows: List[int] = [1, 3, 5]  # 多尺度窗口大小
    smoothing_sigma: float = 1.0  # 相似度平滑参数
    enable_structure_hints: bool = True  # 启用结构提示检测
    enable_hierarchical_output: bool = True  # 启用层次化输出
    
    # 文件处理配置
    max_file_size_mb: int = 50  # 同步处理最大文件大小
    max_async_file_size_mb: int = 100  # 异步处理最大文件大小
    default_chunk_size: int = 5000  # 默认分块大小
    max_concurrent_tasks: int = 3  # 最大并发任务数
    
    # 性能配置
    batch_size: int = 1
    device: str = "auto"  # auto: 自动选择最佳设备, 也可手动设置 "cpu", "cuda", "mps"
    
    class Config:
        env_file = ".env"
        env_prefix = "SEMANTIC_"


# 全局配置实例
settings = Settings()