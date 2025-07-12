#!/usr/bin/env python3
"""
API客户端使用示例
"""

import requests
import json
import time


class SemanticSegmentationClient:
    """语义分段API客户端"""
    
    def __init__(self, base_url="http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API服务基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def segment_text(self, text, threshold=None, separator=None):
        """
        文本分段
        
        Args:
            text: 待分段文本
            threshold: 分段阈值
            separator: 段落分隔符
            
        Returns:
            分段结果
        """
        data = {"text": text}
        if threshold is not None:
            data["threshold"] = threshold
        if separator is not None:
            data["separator"] = separator
        
        try:
            response = self.session.post(
                f"{self.base_url}/segment",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_config(self):
        """获取配置"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def update_config(self, **kwargs):
        """更新配置"""
        try:
            response = self.session.put(
                f"{self.base_url}/config",
                json=kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def main():
    """主函数"""
    print("=== API客户端使用示例 ===\n")
    
    # 创建客户端
    client = SemanticSegmentationClient()
    
    # 健康检查
    print("1. 健康检查")
    health = client.health_check()
    if "error" in health:
        print(f"   服务不可用: {health['error']}")
        print("   请确保API服务正在运行 (python main.py)")
        return
    else:
        print(f"   服务状态: {health['status']}")
        print(f"   版本: {health['version']}")
        print(f"   运行时间: {health['uptime']:.1f}秒")
    
    print()
    
    # 获取配置
    print("2. 获取当前配置")
    config = client.get_config()
    if "error" not in config:
        print(f"   当前阈值: {config['current_config']['threshold']}")
        print(f"   设备: {config['current_config']['device']}")
    
    print()
    
    # 测试文本分段
    print("3. 文本分段测试")
    test_text = """
    深度学习是机器学习的一个重要分支。它通过模拟人脑神经网络的结构和功能来处理数据。
    深度学习在图像识别、自然语言处理、语音识别等领域都取得了突破性进展。
    目前，深度学习已经成为人工智能领域最热门的技术之一。许多大型科技公司都在大力投资这一技术。
    然而，深度学习也面临着一些挑战，比如需要大量的数据和计算资源。
    此外，深度学习模型的可解释性也是一个重要问题。
    """
    
    start_time = time.time()
    result = client.segment_text(test_text.strip())
    end_time = time.time()
    
    if "error" in result:
        print(f"   分段失败: {result['error']}")
    else:
        print(f"   分段成功!")
        print(f"   处理时间: {result['processing_time']:.3f}秒")
        print(f"   段落数: {len(result['paragraphs'])}")
        print(f"   质量分数: {result['quality']['quality_score']:.3f}")
        
        print("\n   分段结果:")
        for i, paragraph in enumerate(result["paragraphs"], 1):
            print(f"     段落{i}: {paragraph}")
    
    print()
    
    # 测试不同阈值
    print("4. 不同阈值测试")
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        result = client.segment_text(test_text.strip(), threshold=threshold)
        if "error" not in result:
            paragraph_count = len(result["paragraphs"])
            quality_score = result["quality"]["quality_score"]
            print(f"   阈值 {threshold}: {paragraph_count}段, 质量分数 {quality_score:.3f}")
    
    print()
    
    # 测试配置更新
    print("5. 配置更新测试")
    update_result = client.update_config(threshold=0.6)
    if "error" not in update_result:
        print(f"   配置更新成功: {update_result['message']}")
        print(f"   新阈值: {update_result['current_config']['threshold']}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()