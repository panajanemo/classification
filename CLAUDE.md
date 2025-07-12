# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于BERT的语义分段服务项目，支持文本和Markdown文件的智能语义分段。项目使用Python开发，集成BERT模型进行语义分析，并提供REST API服务和Web图形界面。

## 核心架构

### 三层架构设计
1. **API层** (`src/api/`): FastAPI应用，包含REST端点、Web界面和文件上传处理
2. **核心处理层** (`src/core/`): BERT模型、语义分段器和分块处理引擎
3. **工具层** (`src/utils/`): 文本处理和Markdown解析工具

### 关键组件交互
- `SemanticSegmenter` 是核心分段器，依赖 `BERTModel` 和 `TextProcessor`
- `ChunkProcessor` 处理大文件的分块和异步任务管理
- `MarkdownProcessor` 解析和重构Markdown文档结构
- 所有组件通过 `config/settings.py` 统一配置

## 设备支持

项目支持自动设备检测和优化:
- **auto**: 自动选择最佳设备 (CUDA > MPS > CPU)
- **mps**: Mac Metal Performance Shaders 加速
- **cuda**: NVIDIA GPU 加速
- **cpu**: CPU 处理

使用 `python check_device.py` 检测设备能力。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动完整服务 (真实BERT模型)
python main.py

# 启动演示服务 (模拟BERT模型，快速启动)
python quick_start.py

# 服务访问
# Web界面: http://localhost:8000/upload
# API文档: http://localhost:8000/docs

# 测试
pytest tests/                           # 运行所有测试
pytest tests/test_markdown_processor.py # 运行特定测试
pytest tests/test_performance.py -v     # 性能测试

# 快速功能验证
python test_service.py                  # 基础功能测试
python test_mps.py                      # Mac MPS配置测试

# 代码质量
black src/                              # 代码格式化
flake8 src/                            # 代码检查

# Docker部署
docker build -t semantic-segmentation .
docker-compose up -d
```

## API端点结构

### 文本处理
- `POST /segment`: 原始文本语义分段
- `POST /upload-markdown`: 同步Markdown文件处理 (≤50MB)
- `POST /upload-markdown-async`: 异步处理 (≤100MB)

### 任务管理
- `GET /task-status/{task_id}`: 查询处理状态
- `GET /task-result/{task_id}`: 获取处理结果
- `GET /tasks`: 列出所有任务

### 系统管理
- `GET /health`: 服务健康检查
- `GET /config`: 获取当前配置
- `PUT /config`: 动态更新配置

## 配置系统

使用环境变量或 `.env` 文件配置，前缀 `SEMANTIC_`:

```bash
SEMANTIC_DEVICE=mps                     # 设备类型
SEMANTIC_THRESHOLD=0.5                  # 分段阈值
SEMANTIC_MODEL_NAME=dennlinger/bert-wiki-paragraphs
SEMANTIC_MAX_FILE_SIZE_MB=50            # 最大文件大小
```

关键配置位于 `config/settings.py`，支持运行时动态调整。

## 文件处理流程

### Markdown处理管道
1. **解析阶段**: `MarkdownProcessor.parse_markdown()` 将文档解析为结构化块
2. **分段阶段**: `segment_markdown_blocks()` 对长段落进行语义分段
3. **重构阶段**: `reconstruct_markdown()` 保持原有格式重新组装

### 大文件处理策略
- 文件 ≤ chunk_size: 直接处理
- 文件 > chunk_size: 分块并行处理 (`ChunkProcessor`)
- 支持异步任务和进度跟踪

## 测试架构

测试分为五个维度:
- `test_text_processor.py`: 文本预处理功能
- `test_semantic_segmentation.py`: 核心分段算法 (使用模拟BERT)
- `test_markdown_processor.py`: Markdown解析和处理
- `test_api.py`: API端点和模型验证
- `test_performance.py`: 性能和并发测试

使用 `tests/conftest.py` 提供测试夹具和模拟数据。

## 部署注意事项

- BERT模型首次加载需要下载约500MB数据
- MPS设备需要macOS 12.3+ 和 Apple Silicon
- 生产环境建议使用 `docker-compose.yml` 部署
- 默认端口8000，支持CORS跨域访问