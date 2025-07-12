# 增强版语义分段服务

基于Sentence Transformer的智能语义分段服务，专为RAG场景优化，能够将长文本自动分割成语义连贯、逻辑完整的段落。

## 功能特点

- 🧠 **智能分段**: 基于Sentence Transformer进行语义分析，精准识别段落边界
- 🔍 **文本类型检测**: 自动识别技术文档、小说、学术论文、新闻、对话等文本类型
- 📏 **多尺度分析**: 采用多尺度窗口计算语义相似度，提高分段精度
- ⚡ **动态阈值**: 根据文本类型自动调整分段阈值，提升适应性
- 🏗️ **层次化输出**: 提供包含元数据的段落结构，支持关键词提取
- 🚀 **高性能**: MPS/GPU加速，支持并发处理和缓存优化
- 📡 **双API接口**: 同时提供标准接口和增强接口
- 🐳 **容器化**: Docker支持，便于部署
- 🔍 **中文优化**: 针对中文文本进行深度优化

## 快速开始

### 环境要求

- Python 3.8+
- torch >= 1.9.0
- sentence-transformers >= 2.2.0
- scipy >= 1.9.0
- FastAPI >= 0.100.0

### 安装依赖

```bash
pip install -r requirements.txt
```

### 支持的设备

- **CPU**: 通用CPU处理，适合开发和轻量部署
- **CUDA**: NVIDIA GPU加速，适合高并发生产环境
- **MPS**: Apple Silicon Mac金属加速，性能优异

### 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

### API文档

启动服务后，访问 `http://localhost:8000/docs` 查看完整的API文档。

## 使用示例

### Python SDK

```python
from src.core.enhanced_semantic_segmenter import EnhancedSemanticSegmenter
from src.core.sentence_transformer_model import SentenceTransformerModel
from src.core.text_type_detector import TextTypeDetector
from src.utils.text_processor import TextProcessor

# 初始化增强版组件
model = SentenceTransformerModel()
text_processor = TextProcessor()
type_detector = TextTypeDetector()
segmenter = EnhancedSemanticSegmenter(model, text_processor, type_detector)

# 执行增强版分段
text = "你的长文本内容..."
result = segmenter.segment_text_enhanced(text)

print(f"检测到文本类型: {result['text_type']}")
print(f"分段质量评分: {result['quality']['quality_score']}")
print("分段结果:")
for i, paragraph in enumerate(result["paragraphs"], 1):
    print(f"段落{i} ({paragraph['type']}): {paragraph['text']}")
    print(f"  关键词: {paragraph['key_phrases']}")
```

### REST API

```bash
# 标准文本分段（使用增强算法）
curl -X POST "http://localhost:8000/segment" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "这是第一句话。这是第二句话，内容相关。这是第三句话，讨论不同主题。",
       "threshold": 0.5
     }'

# 增强版分段（含文本类型检测和层次化输出）
curl -X POST "http://localhost:8000/segment-enhanced" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "你的长文本内容...",
       "threshold": 0.6,
       "enable_auto_threshold": true,
       "force_text_type": "novel",
       "enable_hierarchical_output": true
     }'

# 健康检查
curl http://localhost:8000/health

# 获取配置
curl http://localhost:8000/config
```

### Python客户端

```python
import requests

# 发送分段请求
response = requests.post("http://localhost:8000/segment", json={
    "text": "你的文本内容...",
    "threshold": 0.5
})

result = response.json()
print(f"分段成功，共{len(result['paragraphs'])}段")
```

## 配置说明

### 环境变量

```bash
SEMANTIC_DEBUG=false          # 调试模式
SEMANTIC_HOST=0.0.0.0        # 服务主机
SEMANTIC_PORT=8000           # 服务端口
SEMANTIC_ENHANCED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # Sentence Transformer模型
SEMANTIC_THRESHOLD=0.5       # 分段阈值
SEMANTIC_DEVICE=auto         # 计算设备 (auto/cpu/cuda/mps)
SEMANTIC_USE_ENHANCED_SEGMENTER=true  # 启用增强版分段器
SEMANTIC_ENABLE_AUTO_THRESHOLD=true   # 启用自动阈值调整
```

### 配置文件

复制 `.env.example` 到 `.env` 并修改配置:

```bash
cp .env.example .env
```

## 核心参数

### 分段控制
- **threshold**: 分段阈值 (0.0-1.0)，值越小分段越细
- **enable_auto_threshold**: 启用自动阈值调整
- **multi_scale_windows**: 多尺度窗口大小 [1, 3, 5]
- **min_paragraph_length**: 最小段落长度
- **max_paragraph_length**: 最大段落长度

### 文本类型
- **force_text_type**: 强制指定文本类型 (technical/novel/academic/news/dialogue)
- **enable_structure_hints**: 启用结构提示检测
- **enable_hierarchical_output**: 启用层次化输出

### 性能优化
- **device**: 计算设备选择 (auto/cpu/cuda/mps)
- **batch_size**: 批处理大小
- **cache_enabled**: 启用嵌入缓存

## Docker部署

### 构建镜像

```bash
docker build -t semantic-segmentation .
```

### 运行容器

```bash
docker run -p 8000:8000 semantic-segmentation
```

### Docker Compose

```bash
docker-compose up -d
```

## 开发指南

### 项目结构

```
├── src/
│   ├── api/           # API接口层
│   ├── core/          # 核心算法
│   └── utils/         # 工具模块
├── tests/             # 测试代码
├── config/            # 配置管理
├── examples/          # 使用示例
├── main.py            # 启动脚本
└── requirements.txt   # 依赖文件
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_semantic_segmentation.py

# 运行性能测试
pytest tests/test_performance.py -v
```

### 代码检查

```bash
# 代码格式化
black src/

# 代码检查
flake8 src/

# 类型检查
mypy src/
```

## API参考

### POST /segment

分段文本

**请求参数:**

```json
{
  "text": "待分段的文本内容",
  "threshold": 0.5,
  "separator": "\\n\\n"
}
```

**响应:**

```json
{
  "success": true,
  "paragraphs": ["段落1", "段落2"],
  "formatted_text": "段落1\\n\\n段落2",
  "sentence_count": 4,
  "boundary_count": 1,
  "quality": {
    "paragraph_count": 2,
    "avg_length": 45.5,
    "quality_score": 0.85
  },
  "processing_time": 0.32
}
```

### GET /health

健康检查

### GET /config

获取当前配置

### PUT /config

更新配置参数

## 性能指标

- 单次请求响应时间: < 800ms (Sentence Transformer)
- 内存占用: 600-800MB (含增强功能)
- 支持10+并发处理
- CPU/GPU/MPS全平台兼容
- 缓存命中率: > 85%
- 文本类型检测准确率: > 90%

## 算法原理

### 增强版分段流程

1. **智能预处理**: 中文分句、文本清洗和标准化
2. **文本类型检测**: 自动识别文本类型，选择最优分段策略
3. **语义编码**: 使用Sentence Transformer生成高质量句子向量
4. **多尺度分析**: 计算1句、3句、5句不同窗口的语义相似度
5. **智能边界检测**: 结合多尺度结果和文本类型特征识别边界
6. **层次化构建**: 构建包含元数据的段落结构
7. **质量评估**: 多维度评估分段质量和语义一致性

### 技术创新点

- **多尺度语义分析**: 解决单一尺度分析的局限性
- **文本类型自适应**: 不同类型文本采用不同分段策略
- **动态阈值调整**: 根据文本特征自动优化分段参数
- **层次化输出**: 提供丰富的段落元信息

## 常见问题

### Q: 如何调整分段粒度？

A: 通过 `threshold` 参数控制，值越小分段越细，值越大分段越粗。

### Q: 支持哪些语言？

A: 主要针对中文优化，同时支持英文和中英文混合文本。

### Q: 如何提高性能？

A: 
- 使用GPU加速 (`SEMANTIC_DEVICE=cuda`)
- 调整 `batch_size` 参数
- 使用更快的BERT模型

### Q: 模型加载失败怎么办？

A: 检查网络连接，确保能够访问Hugging Face模型库，或使用本地模型路径。

## 更新日志

### v0.1.0
- 基础语义分段功能
- REST API接口
- Docker支持
- 完整测试覆盖

## 贡献指南

欢迎贡献代码！请先阅读贡献指南并提交Pull Request。

## 许可证

MIT License

## 联系我们

如有问题或建议，请提交Issue或联系开发团队。