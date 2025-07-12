import pytest
from fastapi.testclient import TestClient
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def test_text():
    """测试用中文文本"""
    return """
    人工智能是计算机科学的一个重要分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自从人工智能诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    """


@pytest.fixture
def sample_paragraphs():
    """测试用段落数据"""
    return [
        "这是第一段测试文本。它包含了一些基本的中文内容。",
        "这是第二段测试文本。它与第一段在语义上有所不同。",
        "第三段继续描述不同的主题。每段都有独特的语义特征。"
    ]