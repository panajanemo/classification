# BERT语义分段服务

基于BERT模型的智能语义分段服务，能够将长文本自动分割成语义连贯、逻辑完整的段落。

## 功能特点

- 🧠 **智能分段**: 基于BERT模型进行语义分析，准确识别段落边界
- 🚀 **高性能**: 优化的推理流程，支持并发处理
- 🔧 **可配置**: 支持阈值调整、参数定制
- 📡 **REST API**: 完整的Web API接口
- 🐳 **容器化**: Docker支持，便于部署
- 📊 **质量评估**: 内置分段质量评估指标
- 🔍 **中文优化**: 针对中文文本进行特别优化

## 快速开始

### 环境要求

- Python 3.8+
- torch >= 1.9.0
- transformers >= 4.20.0

### 安装依赖

```bash
pip install -r requirements.txt
```

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
from src.core.semantic_segmenter import SemanticSegmenter
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor

# 初始化组件
bert_model = BERTModel()
text_processor = TextProcessor()
segmenter = SemanticSegmenter(bert_model, text_processor)

# 执行分段
text = "你的长文本内容..."
result = segmenter.segment_text(text)

print("分段结果:")
for i, paragraph in enumerate(result["paragraphs"], 1):
    print(f"段落{i}: {paragraph}")
```

### REST API

```bash
# 文本分段
curl -X POST "http://localhost:8000/segment" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "这是第一句话。这是第二句话，内容相关。这是第三句话，讨论不同主题。",
       "threshold": 0.5
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
SEMANTIC_MODEL_NAME=dennlinger/bert-wiki-paragraphs  # BERT模型
SEMANTIC_THRESHOLD=0.5       # 分段阈值
SEMANTIC_DEVICE=cpu          # 计算设备
```

### 配置文件

复制 `.env.example` 到 `.env` 并修改配置:

```bash
cp .env.example .env
```

## 核心参数

- **threshold**: 分段阈值 (0.0-1.0)，值越小分段越细
- **min_paragraph_length**: 最小段落长度
- **max_paragraph_length**: 最大段落长度
- **separator**: 段落间分隔符

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

- 单次请求响应时间: < 500ms
- 内存占用: 400-500MB (BERT-base)
- 支持并发处理
- CPU/GPU兼容

## 算法原理

1. **文本预处理**: 中文分句、文本清洗和标准化
2. **语义编码**: 使用BERT模型对句子进行编码
3. **相似度计算**: 计算相邻句子的语义相似度
4. **边界识别**: 基于阈值识别段落边界
5. **段落构建**: 动态组合句子构成语义完整段落
6. **质量评估**: 多维度评估分段质量

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