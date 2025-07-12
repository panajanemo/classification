# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于BERT的语义分段服务项目，旨在将长文本智能地分割成语义完整、连贯的段落。项目使用Python开发，集成BERT模型进行语义分析，并提供REST API服务。

## 项目架构

项目采用模块化设计：

### 核心模块
- **文本预处理模块**: 负责中文分句处理
- **BERT模型集成**: 加载和使用dennlinger/bert-wiki-paragraphs模型
- **语义分段算法**: 实现续写概率判断和段落边界识别
- **段落构建器**: 动态组合句子成语义完整段落

### API服务层
- 基于FastAPI的REST API
- 输入输出数据验证
- 错误处理和异常管理
- 配置参数支持（续写概率阈值等）

## 技术栈

- **语言**: Python
- **核心依赖**: transformers (Hugging Face), nltk/jieba (分句), FastAPI (API服务)
- **模型**: dennlinger/bert-wiki-paragraphs 或兼容的BERT模型
- **部署**: Docker容器化

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行API服务
python main.py

# 运行测试
pytest tests/

# 运行单个测试
pytest tests/test_semantic_segmentation.py

# 代码检查
flake8 src/
black src/

# 构建Docker镜像
docker build -t semantic-segmentation .

# 运行Docker容器
docker run -p 8000:8000 semantic-segmentation
```

## 关键配置

- `threshold`: 续写概率阈值，控制分段粒度
- `model_name`: BERT模型名称，支持模型替换
- `separator`: 段落间分隔符配置

## 性能要求

- 单次请求响应时间: < 500ms
- 内存占用: BERT-base约400-500MB
- 支持并发请求处理