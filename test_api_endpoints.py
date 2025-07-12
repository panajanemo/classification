#!/usr/bin/env python3
"""
测试API端点（使用原有的8000端口服务）
"""

import requests
import json


def test_original_endpoint():
    """测试原始分段端点"""
    print("🔍 测试原始分段端点 /segment...")
    
    url = "http://localhost:8000/segment"
    data = {
        "text": "Python是一种强大的编程语言。它具有简洁的语法和丰富的库。机器学习是人工智能的重要分支。深度学习使用神经网络进行学习。BERT模型在自然语言处理中表现出色。",
        "threshold": 0.5
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ 原始端点工作正常")
            print(f"   段落数量: {len(result.get('paragraphs', []))}")
            print(f"   处理时间: {result.get('processing_time', 0):.3f}秒")
            for i, para in enumerate(result.get('paragraphs', []), 1):
                print(f"   段落{i}: {para[:50]}...")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 连接错误: {e}")


def test_enhanced_endpoint():
    """测试增强版分段端点（如果可用）"""
    print("\n🚀 测试增强版分段端点 /segment-enhanced...")
    
    url = "http://localhost:8000/segment-enhanced"
    data = {
        "text": "Python是一种强大的编程语言。它具有简洁的语法和丰富的库。机器学习是人工智能的重要分支。深度学习使用神经网络进行学习。BERT模型在自然语言处理中表现出色。",
        "threshold": 0.5,
        "enable_auto_threshold": True,
        "enable_structure_hints": True,
        "enable_hierarchical_output": True
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ 增强版端点工作正常")
            print(f"   检测文本类型: {result.get('text_type', 'unknown')}")
            print(f"   类型置信度: {result.get('type_confidence', 0):.3f}")
            print(f"   段落数量: {len(result.get('paragraphs', []))}")
            print(f"   处理时间: {result.get('processing_time', 0):.3f}秒")
            
            for i, para in enumerate(result.get('paragraphs', []), 1):
                print(f"   段落{i} [{para.get('type', 'content')}]: {para.get('text', '')[:50]}...")
                if para.get('key_phrases'):
                    print(f"      关键词: {', '.join(para['key_phrases'])}")
                    
        elif response.status_code == 503:
            print("ℹ️  增强版分段器未启用或未初始化完成")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 连接错误: {e}")


def test_api_info():
    """测试API信息"""
    print("\n📋 获取API信息...")
    
    try:
        response = requests.get("http://localhost:8000/api")
        if response.status_code == 200:
            result = response.json()
            print("✅ API信息获取成功")
            print(f"   服务版本: {result.get('version', 'unknown')}")
            endpoints = result.get('endpoints', {})
            print("   可用端点:")
            for name, path in endpoints.items():
                print(f"     {name}: {path}")
        else:
            print(f"❌ 获取API信息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接错误: {e}")


if __name__ == "__main__":
    print("🧪 开始API端点测试...\n")
    
    test_api_info()
    test_original_endpoint()
    test_enhanced_endpoint()
    
    print("\n✅ API端点测试完成！")